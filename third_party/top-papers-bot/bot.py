#!/usr/bin/env python3

import os
import re
import json
import time
import nest_asyncio
import asyncio
from datetime import datetime, timedelta, date, timezone
from typing import Optional, Tuple, List, Dict, Union

import logging
logging.basicConfig(level=logging.INFO)

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters
)

import arxiv
from Bio import Entrez
from libgen_api_enhanced import LibgenSearch
from urllib.error import HTTPError
import xml.etree.ElementTree as ET
import aiohttp

try:
    import requests.utils
except ImportError:
    logging.warning("requests library not found, some URL quoting might not work as expected.")
    import urllib.parse
    class requests:
        class utils:
            @staticmethod
            def quote(s, safe=''):
                return urllib.parse.quote(s, safe=safe)

from tqdm import tqdm
import asyncpg

SEM = asyncio.Semaphore(10)
PG_DSN = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mydatabase")
DB_POOL: Optional[asyncpg.Pool] = None

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from pytz import utc
    SCHEDULER_AVAILABLE = True
    logging.info("APScheduler и pytz успешно импортированы.")
except ImportError as e:
    logging.warning(f"apscheduler или pytz не установлены: {e}. Установите их для фоновой рассылки.")
    AsyncIOScheduler = None
    utc = None
    SCHEDULER_AVAILABLE = False

ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "example@example.com")
Entrez.email = ENTREZ_EMAIL

nest_asyncio.apply()

from g4f.client import AsyncClient
try:
    from g4f.errors import ResponseStatusError, RateLimitError
except ImportError:
    ResponseStatusError = Exception
    RateLimitError = Exception

client_g4f = AsyncClient()

async def init_db():
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = await asyncpg.create_pool(dsn=PG_DSN, min_size=1, max_size=5)
        logging.info("Подключение к PostgreSQL установлено.")

async def get_subscriptions_db() -> List[Dict]:
    global DB_POOL
    if DB_POOL is None:
        await init_db()

    async with DB_POOL.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM subscriptions")
        results = []
        for r in rows:
            results.append(dict(r))
        return results

async def update_subscription_db(chat_id: int, new_data: Dict):
    global DB_POOL
    if DB_POOL is None:
        await init_db()

    keywords = new_data.get("keywords", "")
    field_of_study = new_data.get("field_of_study", "общие науки")
    subscription_period = new_data.get("subscription_period", 0)
    
    raw_lu = new_data.get("last_update")
    if isinstance(raw_lu, datetime):
        lu_for_db = raw_lu.date()
    elif isinstance(raw_lu, date):
        lu_for_db = raw_lu
    elif isinstance(raw_lu, str):
        try:
            lu_for_db = datetime.strptime(raw_lu, "%Y-%m-%d").date()
        except ValueError:
            lu_for_db = datetime(1970, 1, 1).date()
    else:
        lu_for_db = datetime(1970, 1, 1).date()

    max_results = new_data.get("max_results", 20)
    days = new_data.get("days", 0)
    review_only = new_data.get("review_only", False)
    keyword_mode = new_data.get("keyword_mode", "or")
    detailed_query = new_data.get("detailed_query", "")
    min_citations = new_data.get("min_citations", 0)
    databases = new_data.get("databases", "arxiv_pubmed_libgen")
    subscription_hour = new_data.get("subscription_hour", 0)
    subscription_minute = new_data.get("subscription_minute", 0)

    query = """
    INSERT INTO subscriptions
    (chat_id, keywords, field_of_study, subscription_period, last_update, 
     max_results, days, review_only, keyword_mode, detailed_query, 
     min_citations, databases, subscription_hour, subscription_minute)
    VALUES ($1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13, $14)
    ON CONFLICT (chat_id)
    DO UPDATE SET
        keywords = EXCLUDED.keywords,
        field_of_study = EXCLUDED.field_of_study,
        subscription_period = EXCLUDED.subscription_period,
        last_update = EXCLUDED.last_update,
        max_results = EXCLUDED.max_results,
        days = EXCLUDED.days,
        review_only = EXCLUDED.review_only,
        keyword_mode = EXCLUDED.keyword_mode,
        detailed_query = EXCLUDED.detailed_query,
        min_citations = EXCLUDED.min_citations,
        databases = EXCLUDED.databases,
        subscription_hour = EXCLUDED.subscription_hour,
        subscription_minute = EXCLUDED.subscription_minute
    """

    async with DB_POOL.acquire() as conn:
        await conn.execute(query,
            chat_id,
            keywords,
            field_of_study,
            subscription_period,
            lu_for_db,
            max_results,
            days,
            review_only,
            keyword_mode,
            detailed_query,
            min_citations,
            databases,
            subscription_hour,
            subscription_minute
        )

async def remove_subscription_db(chat_id: int):
    global DB_POOL
    if DB_POOL is None:
        await init_db()

    async with DB_POOL.acquire() as conn:
        await conn.execute("DELETE FROM subscriptions WHERE chat_id=$1", chat_id)

def parse_datetime_from_string(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return datetime(1970, 1, 1)

def check_review_keywords(text: str) -> bool:
    keywords = [
        "review", "survey", "overview", "comprehensive",
        "state-of-the-art", "literature review", "systematic review"
    ]
    return any(word in text.lower() for word in keywords)

def match_keywords(paper: Dict, keywords: List[str], mode: str) -> bool:
    content = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    if mode.lower() == "and":
        return all(keyword.lower() in content for keyword in keywords)
    return any(keyword.lower() in content for keyword in keywords)

def time_to_send(last_update_val: Union[str, date, datetime], period_days: int) -> bool:
    if period_days <= 0:
        return False
    
    last_update_dt: datetime
    if isinstance(last_update_val, str):
        last_update_dt = parse_datetime_from_string(last_update_val) 
    elif isinstance(last_update_val, date): 
        last_update_dt = datetime.combine(last_update_val, datetime.min.time())
    elif isinstance(last_update_val, datetime): 
        last_update_dt = last_update_val
    else: 
        last_update_dt = datetime(1970, 1, 1)

    today_utc = datetime.now(timezone.utc).date()
    last_update_date = last_update_dt.date()
    
    days_passed = (today_utc - last_update_date).days
    logging.debug(f"time_to_send: последнее обновление {last_update_date}, сегодня {today_utc}, прошло дней: {days_passed}, период: {period_days}")
    
    return days_passed >= period_days

async def check_subscriptions(application: Application):
    """Улучшенная функция проверки подписок с отправкой уведомлений об отсутствии статей"""
    logging.info("=== Запуск фоновой проверки подписок (UTC) ===")
    
    try:
        subs = await get_subscriptions_db()
        logging.info(f"Найдено подписок: {len(subs)}")
    except Exception as e:
        logging.error(f"Ошибка при получении подписок из БД: {e}")
        return
    
    now_utc = datetime.now(timezone.utc)
    logging.info(f"Текущее время UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    for sub_original in subs: 
        chat_id = sub_original["chat_id"]
        period = sub_original.get("subscription_period", 0)
        last_update_from_db = sub_original.get("last_update", datetime(1970, 1, 1).date())
        sub_hour = sub_original.get("subscription_hour", 0)
        sub_minute = sub_original.get("subscription_minute", 0)

        logging.info(f"Проверяем подписку chat_id={chat_id}: период={period} дн., последнее обновление={last_update_from_db}, время рассылки={sub_hour:02d}:{sub_minute:02d} UTC")

        if not should_send_subscription_now(last_update_from_db, period, sub_hour, sub_minute):
            logging.debug(f"Подписка chat_id={chat_id}: еще не время для рассылки.")
            continue
        
        logging.info(f"Подписка chat_id={chat_id} готова к выполнению")

        date_from_for_search: datetime
        if isinstance(last_update_from_db, date): 
            date_from_for_search = datetime.combine(last_update_from_db, datetime.min.time())
        elif isinstance(last_update_from_db, datetime): 
             date_from_for_search = last_update_from_db
        elif isinstance(last_update_from_db, str): 
             date_from_for_search = parse_datetime_from_string(last_update_from_db)
        else: 
             date_from_for_search = datetime(1970, 1, 1)

        user_data_for_search = { 
            "keywords": sub_original.get("keywords", ""),
            "max_results": sub_original.get("max_results", 20),
            "days": 0, 
            "review_only": sub_original.get("review_only", False),
            "keyword_mode": sub_original.get("keyword_mode", "or"),
            "detailed_query": sub_original.get("detailed_query", "") or sub_original.get("keywords", ""),
            "field_of_study": sub_original.get("field_of_study", "общие науки"),
            "min_citations": sub_original.get("min_citations", 0),
            "databases": sub_original.get("databases", "arxiv_pubmed_libgen"),
        }

        logging.info(f"Выполняем поиск для chat_id={chat_id} с даты {date_from_for_search.strftime('%Y-%m-%d')}")
        
        papers = []
        search_successful = True
        try:
            papers = await perform_search_and_ranking(user_data_for_search, date_from=date_from_for_search)
        except Exception as e:
            logging.error(f"Ошибка при поиске для chat_id={chat_id}: {e}", exc_info=True)
            search_successful = False

        data_for_db_update = sub_original.copy() 
        data_for_db_update["last_update"] = now_utc.date()

        bot = application.bot

        if not search_successful:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text="⚠️ Произошла ошибка при поиске новых статей по вашей подписке. "
                         "Попробуем снова при следующей проверке.",
                    parse_mode="HTML"
                )
            except Exception as e:
                logging.error(f"Не удалось отправить сообщение об ошибке пользователю {chat_id}: {e}")
            
            await update_subscription_db(chat_id, data_for_db_update)
            continue

        if not papers:
            try:
                keywords = user_data_for_search.get("keywords", "ваш запрос")
                period_text = f"{period} день" if period == 1 else f"{period} дня" if period < 5 else f"{period} дней"
                
                no_updates_message = (
                    f"📬 Обновление подписки\n\n"
                    f"🔍 Запрос: <b>{keywords}</b>\n"
                    f"📅 Период: за последние {period_text}\n\n"
                    f"🤷‍♂️ К сожалению, новых статей по вашему запросу за указанный период не найдено.\n\n"
                    f"💡 Попробуйте:\n"
                    f"• Расширить ключевые слова\n"
                    f"• Изменить период подписки\n"
                    f"• Проверить настройки фильтров\n\n"
                    f"Следующая проверка: через {period_text}"
                )
                
                await bot.send_message(
                    chat_id=chat_id,
                    text=no_updates_message,
                    parse_mode="HTML"
                )
                logging.info(f"Отправлено уведомление об отсутствии статей для chat_id={chat_id}")
            except Exception as e:
                logging.error(f"Не удалось отправить уведомление об отсутствии статей пользователю {chat_id}: {e}")
            
            await update_subscription_db(chat_id, data_for_db_update)
            continue

        messages_sent = 0
        update_last_update = True        

        try:
            keywords = user_data_for_search.get("keywords", "ваш запрос")
            period_text = f"{period} день" if period == 1 else f"{period} дня" if period < 5 else f"{period} дней"
            
            header_message = (
                f"📬 <b>Новые статьи по подписке!</b>\n\n"
                f"🔍 Запрос: <b>{keywords}</b>\n"
                f"📅 Период: за последние {period_text}\n"
                f"📊 Найдено статей: <b>{len(papers)}</b>\n\n"
                f"📚 Вот что мы нашли для вас:"
            )
            
            await bot.send_message(
                chat_id=chat_id,
                text=header_message,
                parse_mode="HTML"
            )
        except Exception as e:
            logging.error(f"Не удалось отправить заголовок рассылки пользователю {chat_id}: {e}")
        
        for paper in papers: 
            try:
                authors_text = paper.get("authors") or "Авторы не указаны"
                title = paper.get("title", "Без названия")
                date_p_str = paper.get("publication_date", "N/A") 
                cites = paper.get("cited_by", "N/A")
                rating = paper.get("rating", 0)
                doi_or_id = paper.get("doi", "") or paper.get("paperId", "") 
                source = paper.get("source", "Неизвестный источник")
                abstract = paper.get("abstract") or "Нет аннотации"

                link = "N/A"
                if source == "arXiv":
                    paper_id_val = paper.get("paperId") or doi_or_id.replace("arxiv:", "")
                    link = f"https://arxiv.org/abs/{paper_id_val}"
                elif source == "PMC":
                    pmcid = paper.get("paperId") or doi_or_id
                    if pmcid and not pmcid.startswith("PMC"): 
                        pmcid = "PMC" + pmcid.replace("PMC", "")
                    elif not pmcid: 
                        pmcid = "N/A"
                    link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}" if pmcid != "N/A" else "N/A"
                elif source == "Libgen":
                    quoted_title = requests.utils.quote(paper.get('title', ''))
                    dl_links = paper.get("download_links", {})
                    if isinstance(dl_links, dict) and dl_links.get("direct"):
                        link = dl_links["direct"]
                    elif isinstance(dl_links, dict) and dl_links.get("mirror_1"):
                        link = dl_links["mirror_1"]
                    else:
                        link = f"https://libgen.is/search.php?req={quoted_title}"

                text_html = (
                    f"<b>{title}</b>\n"
                    f"👤 Автор(ы): {authors_text}\n"
                    f"📅 Дата: {date_p_str}\n"
                    f"📊 Цитирований: {cites if cites is not None else 'N/A'}\n"
                    f"⭐ Рейтинг (LLM): {rating}\n"
                    f"🆔 DOI/ID: {doi_or_id}\n"
                    f"📚 Источник: {source}\n"
                    f"🔗 Ссылка: <a href='{link}'>{link}</a>\n\n"
                    f"📜 <i>Аннотация:</i> {abstract[:500]}{'...' if len(abstract) > 500 else ''}"
                )
                
                await bot.send_message(
                    chat_id=chat_id,
                    text=text_html,
                    parse_mode="HTML",
                    disable_web_page_preview=True
                )
                messages_sent += 1
                await asyncio.sleep(0.5) 
                
            except Exception as e:
                from telegram.error import Unauthorized, NetworkError 
                logging.error(f"Ошибка при отправке статьи пользователю {chat_id}: {e}")
                if isinstance(e, Unauthorized):
                    logging.info(f"Пользователь {chat_id} недоступен (Unauthorized). Удаляем подписку.")
                    await remove_subscription_db(chat_id)
                    update_last_update = False
                    break
                elif isinstance(e, NetworkError):
                    logging.warning(f"Сетевая ошибка при отправке {chat_id}. Пропускаем эту статью.")
        
        if update_last_update:
            if messages_sent > 0:
                logging.info(f"Отправлено {messages_sent} новых статей для chat_id={chat_id}")

                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=f"✅ Рассылка завершена! Отправлено статей: {messages_sent}\n"
                             f"Следующая проверка: через {period} дней в {sub_hour:02d}:{sub_minute:02d} UTC",
                        parse_mode="HTML"
                    )
                except Exception as e:
                    logging.error(f"Не удалось отправить сообщение об окончании рассылки пользователю {chat_id}: {e}")
            
            await update_subscription_db(chat_id, data_for_db_update)

    logging.info("=== Завершение фоновой проверки подписок (UTC) ===")


async def get_citation_count(doi: str) -> int:
    async with SEM:
        if doi.startswith("arxiv:"):
            paper_id = "ARXIV:" + doi.split("arxiv:")[1]
        elif doi.startswith("PMC:"):
            return 0
        elif doi.startswith("10."):
            paper_id = "DOI:" + doi
        else:
            paper_id = doi

        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=citationCount"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("citationCount", 0)
                    else:
                        return 0
        except Exception as e:
            print(f"Error fetching citation count for {doi}: {e}")
            return 0

async def get_references(doi: str) -> List[Dict]:
    async with SEM:
        if doi.startswith("arxiv:"):
            paper_id = "ARXIV:" + doi.split("arxiv:")[1]
        elif doi.startswith("PMC:"):
            return []
        elif doi.startswith("10."):
            paper_id = "DOI:" + doi
        else:
            paper_id = doi

        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=references.title,references.paperId"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("references", [])
                    else:
                        return []
        except Exception as e:
            print(f"Error fetching references for {doi}: {e}")
            return []

def arxiv_search(query: str, max_results: int, date_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict]:
    client_arxiv = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search_obj = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results = []
    count = 0

    print("Starting arXiv search...")
    with tqdm(total=max_results, desc="arXiv Search", unit="paper") as pbar:
        try:
            for result in client_arxiv.results(search_obj):
                pub_date_dt = result.published.replace(tzinfo=None) 
                if date_range and not (date_range[0] <= pub_date_dt <= date_range[1]):
                    continue
                
                authors_list = [author.name for author in result.authors]
                authors_str = ", ".join(authors_list) if authors_list else "Авторы не указаны"

                results.append({
                    "title": result.title,
                    "abstract": result.summary,
                    "publication_date": pub_date_dt.strftime("%Y-%m-%d"), 
                    "doi": result.doi or f"arxiv:{result.entry_id.split('/')[-1]}",
                    "paperId": result.entry_id.split('/')[-1],
                    "source": "arXiv",
                    "authors": authors_str,
                    "is_review": check_review_keywords(result.title + " " + result.summary),
                    "cited_by": 0,
                    "references": []
                })

                pbar.update(1)
                count += 1
                if count >= max_results:
                    break
        except arxiv.UnexpectedEmptyPageError:
            logging.warning("arXiv search returned an unexpected empty page.")
        except HTTPError as he:
            logging.error(f"HTTPError during arXiv search: {he}")
        except Exception as e:
            logging.error(f"Generic error during arXiv search: {e}", exc_info=True)

    print(f"arXiv search completed. Found {len(results)} results.")
    return results

def pmc_search(query: str, max_results: int, date_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict]:
    results = []
    print("Starting PMC search...")
    try:
        handle = Entrez.esearch(db="pmc", term=query, retmax=min(max_results, 500), sort='relevance')
        record = Entrez.read(handle, validate=False)
        handle.close()
    except Exception as e:
        print(f"Error during esearch in PMC: {e}")
        return results

    ids = record.get("IdList", [])
    ids_to_fetch = ids[:max_results]

    with tqdm(total=len(ids_to_fetch), desc="PMC Search", unit="paper") as pbar:
        for pmc_id_raw in ids_to_fetch:
            try:
                pmc_id = pmc_id_raw
                if not pmc_id.startswith("PMC"):
                    pmc_id_query = "PMC" + pmc_id
                else:
                    pmc_id_query = pmc_id
                
                time.sleep(0.35)
                handle_efetch = Entrez.efetch(db="pmc", id=pmc_id_query, retmode="xml")
                xml_data = handle_efetch.read()
                handle_efetch.close()

                if isinstance(xml_data, bytes):
                    xml_data = xml_data.decode('utf-8', errors='ignore')
                if not xml_data.strip():
                    print(f"PMC fetch returned empty result for {pmc_id_query}")
                    pbar.update(1)
                    continue
                
                try:
                    root = ET.fromstring(xml_data)
                except ET.ParseError as pe:
                    print(f"PMC parsing error ({pmc_id_query}): XML Parse Error: {pe}. XML: {xml_data[:200]}... Skipping.")
                    pbar.update(1)
                    continue

                title_elem = root.find('.//article-title')
                title = ''.join(title_elem.itertext()).strip() if title_elem is not None else 'No Title'

                abstract_elem = root.find('.//abstract')
                abstract_parts = []
                if abstract_elem is not None:
                    for sec in abstract_elem.findall('.//sec'):
                        sec_title_elem = sec.find('./title')
                        sec_title = f"{''.join(sec_title_elem.itertext()).strip()}: " if sec_title_elem is not None and sec_title_elem.text else ""
                        p_text = "".join(p_elem.text.strip() for p_elem in sec.findall('./p') if p_elem.text)
                        abstract_parts.append(f"{sec_title}{p_text}")
                    if not abstract_parts:
                         abstract_parts.append("".join(abstract_elem.itertext()).strip())
                abstract = "\n".join(abstract_parts) if abstract_parts else 'No Abstract'

                doi_elem = root.find(".//article-id[@pub-id-type='doi']")
                doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else f"PMC:{pmc_id_query.replace('PMC','')}"

                authors_list = []
                for contrib in root.findall('.//contrib-group/contrib[@contrib-type="author"]'):
                    surname_elem = contrib.find('./name/surname')
                    given_names_elem = contrib.find('./name/given-names')
                    name_str = ""
                    if given_names_elem is not None and given_names_elem.text:
                        name_str += given_names_elem.text
                    if surname_elem is not None and surname_elem.text:
                        name_str += (" " + surname_elem.text) if name_str else surname_elem.text
                    if name_str.strip():
                        authors_list.append(name_str.strip())
                authors_str = ", ".join(authors_list) if authors_list else "Авторы не указаны"

                pub_date_elem = (
                    root.find(".//pub-date[@pub-type='epub']") or
                    root.find(".//pub-date[@pub-type='ppub']") or
                    root.find(".//pub-date")
                )
                year_str, month_str, day_str = "1970", "1", "1"
                if pub_date_elem is not None:
                    year_str = pub_date_elem.findtext("year", default=year_str)
                    month_str = pub_date_elem.findtext("month", default=month_str)
                    day_str = pub_date_elem.findtext("day", default=day_str)
                
                try:
                    pub_date_dt = datetime(int(year_str), int(month_str), int(day_str))
                except ValueError:
                    pub_date_dt = datetime(int(year_str), 1, 1)

                if date_range and not (date_range[0] <= pub_date_dt <= date_range[1]):
                    pbar.update(1)
                    continue

                is_review = check_review_keywords(title + " " + abstract)

                results.append({
                    "title": title,
                    "abstract": abstract,
                    "publication_date": pub_date_dt.strftime('%Y-%m-%d'),
                    "doi": doi,
                    "paperId": pmc_id_query,
                    "source": "PMC",
                    "authors": authors_str,
                    "is_review": is_review,
                    "cited_by": 0,
                    "references": []
                })
            except HTTPError as he:
                print(f"PMC HTTPError ({pmc_id_query}): {he}")
            except Exception as e:
                print(f"PMC general error ({pmc_id_query}): {e}", exc_info=True)

            pbar.update(1)

    print(f"PMC search completed. Found {len(results)} results.")
    return results

def libgen_search_api(query: str, max_results: int, do_get_download_links: bool = False) -> List[Dict]:
    try:
        from libgen_api_enhanced import LibgenSearch as LibgenSearchClient 
    except ImportError:
        logging.warning("libgen_api_enhanced not installed. Libgen search will be skipped.")
        return []

    def is_isbn_code(q: str) -> bool:
        candidate = q.replace("-", "").replace(" ", "")
        return len(candidate) in (10, 13) and candidate.isdigit()

    mirrors = [
        "http://libgen.is",
        "http://libgen.rs",
        "http://libgen.li",
    ]

    results_raw = []
    mirror_used = None

    print("Starting Libgen search...")
    for mirror_url in mirrors:
        try:
            s = LibgenSearchClient()
            s.base_url = mirror_url 
            s.timeout = 20

            if is_isbn_code(query):
                results_raw = s.search_default(query, search_type="isbn") 
            else:
                results_raw = s.search_title_filtered(query, filters={}, exact_match=False)
                if not results_raw or len(results_raw) < 5:
                    author_results = s.search_author_filtered(query, filters={}, exact_match=False)
                    results_raw.extend(author_results) 
                    seen_ids = set()
                    unique_results = []
                    for item in results_raw:
                        if item.get("ID") not in seen_ids:
                            unique_results.append(item)
                            seen_ids.add(item.get("ID"))
                    results_raw = unique_results

            mirror_used = mirror_url
            if results_raw:
                break
        except Exception as e: 
            print(f"Error with LibgenSearch on mirror {mirror_url}: {e}")

    if not results_raw:
        print("No Libgen results found or all mirrors failed.")
        return []

    slice_length = min(max_results, len(results_raw))
    results = []

    print(f"Using mirror: {mirror_used}")
    with tqdm(total=slice_length, desc="Libgen Search", unit="book") as pbar:
        for book in results_raw[:slice_length]:
            authors_str = book.get("Author", "Авторы не указаны")
            if not authors_str or authors_str.isspace():
                authors_str = "Авторы не указаны"

            r = {
                "title": book.get("Title", "Без названия"),
                "abstract": "Аннотация отсутствует.",
                "publication_date": book.get("Year", "N/A"), 
                "doi": f"Libgen_ID:{book.get('ID', '')}",
                "paperId": book.get("ID", ""),
                "source": "Libgen",
                "authors": authors_str,
                "is_review": False,
                "cited_by": None,
                "references": [],
                "publisher": book.get("Publisher", ""),
                "language": book.get("Language", ""),
                "extension": book.get("Extension", ""),
                "cover_url": book.get("Cover", ""),
                "download_links": { 
                    "direct": book.get("Direct_Download_Link", ""),
                    "mirror_1": book.get("Mirror_1", ""),
                    "mirror_2": book.get("Mirror_2", ""),
                    "mirror_3": book.get("Mirror_3", ""),
                } if do_get_download_links else {}, 
            }
            results.append(r)
            pbar.update(1)
            time.sleep(0.01)

    print(f"Libgen search completed. Found {len(results)} results.")
    return results

def search_papers(query: str, 
                  max_results: int = 20, 
                  days: int = 0, 
                  review_only: bool = False, 
                  keyword_mode: str = "or",
                  databases: str = "arxiv_pubmed_libgen",
                  date_from: Optional[datetime] = None) -> List[Dict]: 
    date_range: Optional[Tuple[datetime, datetime]] = None
    
    end_date = datetime.now()
    if days > 0:
        start_date = end_date - timedelta(days=days)
        date_range = (start_date, end_date)
    elif date_from is not None: 
        date_range = (date_from, end_date)

    keywords_list = [k.strip().lower() for k in query.split(",") if k.strip()]
    if not keywords_list:
        keywords_list = [k.lower() for k in query.strip().split()]

    if keywords_list:
        if keyword_mode.lower() == "or":
            api_query_for_academic_dbs = " OR ".join(f'"{k}"' for k in keywords_list)
        else:
            api_query_for_academic_dbs = " AND ".join(f'"{k}"' for k in keywords_list)
    else:
        api_query_for_academic_dbs = query

    libgen_query_str = query

    source_functions = [] 
    if "arxiv" in databases:
        source_functions.append((arxiv_search, api_query_for_academic_dbs))
    if "pubmed" in databases:
        source_functions.append((pmc_search, api_query_for_academic_dbs))
    if "libgen" in databases:
        source_functions.append((libgen_search_api, libgen_query_str)) 

    if not source_functions: 
        source_functions = [
            (arxiv_search, api_query_for_academic_dbs), 
            (pmc_search, api_query_for_academic_dbs), 
            (libgen_search_api, libgen_query_str)
        ]

    num_sources = len(source_functions)
    per_source = max_results // num_sources if num_sources > 0 else max_results
    remainder = max_results % num_sources if num_sources > 0 else 0
    
    results = []

    print("=== Starting unified search across sources ===")
    for i, (source_func, current_query) in enumerate(source_functions):
        current_max_results = per_source + (1 if i < remainder else 0)
        if current_max_results == 0 and max_results > 0: 
            current_max_results = 1

        print(f"Searching in {source_func.__name__} with query: '{current_query}' for max {current_max_results} results...")
        if source_func == libgen_search_api:
            results.extend(source_func(current_query, current_max_results, do_get_download_links=False))
        else:
            results.extend(source_func(current_query, current_max_results, date_range))
        print(f"Done with {source_func.__name__}. {len(results)} total results so far.")

    unique_papers = {}
    for paper in results:
        title_key = paper.get("title", "").strip().lower()
        id_key_part = paper.get("doi", paper.get("paperId", "")).strip().lower()
        
        if "doi:" in id_key_part or "10." in id_key_part:
            unique_key = f"{title_key}|{id_key_part}"
        else:
            unique_key = f"{title_key}|{paper.get('source', '').lower()}|{id_key_part}"

        if unique_key not in unique_papers:
            unique_papers[unique_key] = paper
        else:
            existing_paper = unique_papers[unique_key]
            if paper.get("source") != "Libgen" and existing_paper.get("source") == "Libgen":
                unique_papers[unique_key] = paper
            elif paper.get("abstract") and not existing_paper.get("abstract"):
                 unique_papers[unique_key] = paper

    results = list(unique_papers.values())

    if keywords_list:
        results = [p for p in results if match_keywords(p, keywords_list, keyword_mode)]

    if review_only:
        results = [p for p in results if p.get("is_review")]

    results.sort(key=lambda x: parse_datetime_from_string(x.get("publication_date", "1970-01-01")), reverse=True)
    return results[:max_results]

def save_papers(papers: List[Dict], filename: str = "papers.json"):
    if not isinstance(papers, list):
        raise ValueError("Expected papers to be a list of dictionaries with metadata.")

    formatted_papers = []
    
    for paper in papers:
        formatted = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", "").replace("\n", " ").strip(),
            "publication_date": paper.get("publication_date", ""),
            "doi": paper.get("doi", ""),
            "paperId": paper.get("paperId", ""),
            "source": paper.get("source", ""),
            "authors": paper.get("authors", "Авторы не указаны"),
            "is_review": paper.get("is_review", False),
            "cited_by": paper.get("cited_by", 0), 
            "rating": paper.get("rating", 0)
        }

        references = []
        for ref in paper.get("references", []):
            if isinstance(ref, dict):
                ref_data = {
                    "paperId": ref.get("paperId") or ref.get("arxivId") or ref.get("pmcid") or None,
                    "title": ref.get("title", "").replace("\n", " ").strip() if ref.get("title") else "Без названия"
                }
                if any(ref_data.values()):
                    references.append(ref_data)
        formatted["references"] = references

        if paper.get("source") == "Libgen":
            formatted.update({
                "publisher": paper.get("publisher", ""),
                "language": paper.get("language", ""),
                "extension": paper.get("extension", ""),
                "cover_url": paper.get("cover_url", ""),
                "download_links": paper.get("download_links", {}),
            })

        formatted_papers.append(formatted)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            formatted_papers, 
            f,
            ensure_ascii=False,
            indent=2,
            default=str 
        )

async def evaluate_paper(paper: Dict, detailed_query: str, field_of_study: str, retries: int = 3) -> int:
    async with SEM:
        prompt = f"""You are an expert in {field_of_study}. Rate the relevance (1-100) of the following paper to this query: "{detailed_query}"
Title: {paper.get('title', 'No Title')}
Abstract: {paper.get('abstract', 'No Abstract')}
Answer strictly with an integer."""

        for attempt in range(retries):
            try:
                response = await client_g4f.chat.completions.create(
                    model="deepseek-r1", 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    timeout=20
                )
                rating_text = response.choices[0].message.content.strip()
                match = re.search(r'(\d+)', rating_text)
                if match:
                    rating = int(match.group(1))
                    return min(max(rating, 0), 100)
                else:
                    logging.warning(f"No integer found in LLM response for '{paper.get('title')}': {rating_text}")
            except RateLimitError: 
                logging.warning(f"Rate limit hit for paper '{paper.get('title')}'. Attempt {attempt + 1}/{retries}. Retrying after delay.")
                await asyncio.sleep(10 * (attempt + 1))
            except ResponseStatusError as rse: 
                 logging.error(f"Response status error for paper '{paper.get('title')}': {rse}. Attempt {attempt + 1}/{retries}.")
                 await asyncio.sleep(5 * (attempt + 1))
            except asyncio.TimeoutError:
                 logging.error(f"Timeout error for paper '{paper.get('title')}' on attempt {attempt + 1}/{retries}.")
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{retries} failed for paper '{paper.get('title')}': {e}", exc_info=True)
                await asyncio.sleep(5) 
    return 0 

async def rank_papers(papers: List[Dict], detailed_query: str, field_of_study: str) -> List[Dict]:
    if not detailed_query or detailed_query.strip() == "":
        pass

    tasks = [evaluate_paper(p, detailed_query, field_of_study) for p in papers]
    ratings = await asyncio.gather(*tasks)
    for paper, rating in zip(papers, ratings):
        paper['rating'] = rating
    papers.sort(key=lambda x: x.get('rating', 0), reverse=True)
    return papers

async def perform_search_and_ranking(user_data: Dict, progress_callback=None, date_from: Optional[datetime] = None) -> List[Dict]:
    keywords = user_data.get("keywords", "")
    max_results = user_data.get("max_results", 20)
    days = user_data.get("days", 0) 
    review_only = user_data.get("review_only", False)
    keyword_mode = user_data.get("keyword_mode", "or")
    detailed_query_for_llm = user_data.get("detailed_query", "").strip()
    if not detailed_query_for_llm:
        detailed_query_for_llm = keywords

    field_of_study = user_data.get("field_of_study", "общие науки")
    min_citations = user_data.get("min_citations", 0)
    chosen_databases = user_data.get("databases", "arxiv_pubmed_libgen") 

    if progress_callback:
        await progress_callback("Шаг 1/5: Ищем подходящие статьи в выбранных базах...")

    papers = search_papers(
        query=keywords,
        max_results=max_results,
        days=days, 
        review_only=review_only,
        keyword_mode=keyword_mode,
        databases=chosen_databases, 
        date_from=date_from 
    )
    if not papers:
        return []

    if progress_callback:
        await progress_callback(f"Шаг 2/5: Найдено {len(papers)} статей. Получаем данные о цитированиях...")

    citation_tasks = []
    for paper in papers:
        if paper.get("source") != "Libgen" and paper.get("doi") and ("arxiv" in paper.get("doi").lower() or "10." in paper.get("doi")):
            citation_tasks.append(get_citation_count(paper["doi"]))
        else:
            fut = asyncio.Future()
            fut.set_result(0 if paper.get("source") != "Libgen" else None)
            citation_tasks.append(fut)
    
    citation_counts = await asyncio.gather(*citation_tasks)
    for paper, count in zip(papers, citation_counts):
        paper["cited_by"] = count

    if progress_callback:
        await progress_callback("Шаг 3/5: Получаем информацию о ссылках (references)...")
    
    reference_tasks = []
    for paper in papers:
        if paper.get("source") != "Libgen" and paper.get("doi") and ("arxiv" in paper.get("doi").lower() or "10." in paper.get("doi")):
            reference_tasks.append(get_references(paper["doi"]))
        else:
            fut_refs = asyncio.Future()
            fut_refs.set_result([])
            reference_tasks.append(fut_refs)

    references_results = await asyncio.gather(*reference_tasks)
    for paper, refs in zip(papers, references_results):
        paper["references"] = refs

    if progress_callback:
        await progress_callback("Шаг 4/5: Фильтруем по минимальному числу цитирований...")

    if min_citations > 0:
        papers_before_citation_filter = len(papers)
        papers = [
            p for p in papers
            if p.get("source") == "Libgen"
               or (p.get("cited_by") is not None and p["cited_by"] >= min_citations)
        ]
        if progress_callback:
            await progress_callback(f"Отфильтровано по цитированиям: осталось {len(papers)} из {papers_before_citation_filter} статей.")

    if not papers:
        return []

    if progress_callback:
        await progress_callback("Шаг 5/5: Ранжируем результаты с помощью LLM...")
    
    papers = await rank_papers(papers, detailed_query_for_llm, field_of_study)
    return papers

(
    STATE_DATABASES, 
    STATE_KEYWORDS,
    STATE_KEYWORD_MODE,
    STATE_DETAILED_QUERY,
    STATE_FIELD_OF_STUDY,
    STATE_MAX_RESULTS,
    STATE_MAX_RESULTS_CUSTOM,
    STATE_TOP_COUNT,
    STATE_DAYS,
    STATE_MIN_CITATIONS,
    STATE_REVIEWS_ONLY,
    STATE_PERIODIC_UPDATES,
    STATE_SUBSCRIPTION_HOUR,
    STATE_SUBSCRIPTION_MINUTE,
    STATE_SUBSCRIPTION_MINUTE_CUSTOM,  
    STATE_CONFIRM_PARAMS
) = range(16)  


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    unsubscribe_button = None

    try:
        subscriptions = await get_subscriptions_db()
        if any(sub.get("chat_id") == chat_id for sub in subscriptions):
            unsubscribe_button = InlineKeyboardButton("📤 Отписаться от рассылки", callback_data="unsubscribe")
    except Exception as e:
        logging.error("Ошибка при загрузке подписок: %s", e)

    keyboard_layout = [
        [
            InlineKeyboardButton("🔍 Поиск статей", callback_data="start_search"),
            InlineKeyboardButton("❌ Отмена", callback_data="cancel")
        ]
    ]
    if unsubscribe_button:
        keyboard_layout.append([unsubscribe_button])

    welcome_message = (
        "Привет! 👋 Я бот для поиска научных статей в arXiv, PubMed (PMC) и Libgen. "
        "Я могу не только искать по ключевым словам, но и использовать большую языковую модель (LLM) "
        "для интеллектуального поиска по вашим сложным запросам. "
        "Это позволяет находить более релевантные статьи, особенно если ваш интерес охватывает несколько аспектов или требует глубокого понимания контекста.\n\n"
        "▶️ Нажмите «Поиск статей», чтобы начать.\n"
        "🚫 «Отмена» прервет текущий процесс настройки поиска.\n"
        "📤 Если вы ранее оформляли подписку, появится кнопка «Отписаться от рассылки»."
    )
    await update.message.reply_text(
        welcome_message,
        reply_markup=InlineKeyboardMarkup(keyboard_layout)
    )
    return ConversationHandler.END

async def unsubscribe_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id

    try:
        await remove_subscription_db(chat_id)
        await query.message.reply_text("Вы успешно отписались от рассылки. ✅\n"
                                       "Теперь вы не будете получать периодические обновления по запросу.")
    except Exception as e:
        logging.error("Ошибка при отписке: %s", e)
        await query.message.reply_text("Произошла ошибка при попытке отписаться. Попробуйте позже. ❌")

async def start_search_trigger(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    context.user_data.clear() 
    return await ask_databases_first_step(update.callback_query.message, context)

async def cancel_trigger(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_interface = update.message if update.message else update.callback_query.message
    if update.callback_query:
        await update.callback_query.answer()
    
    await message_interface.reply_text("Текущая операция отменена. Вы всегда можете начать заново, нажав «Поиск статей» или команду /search.")
    context.user_data.clear()
    return ConversationHandler.END

async def start_search_command_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    return await ask_databases_first_step(update.message, context)

async def ask_databases_first_step(message_interface, context: ContextTypes.DEFAULT_TYPE):
    current_selection = context.user_data.get("databases_selection", [])
    
    def btn_text(db_id, text):
        return f"✅ {text}" if db_id in current_selection else text

    keyboard = [
        [
            InlineKeyboardButton(btn_text("arxiv", "arXiv"), callback_data="db_toggle_arxiv_entry"),
            InlineKeyboardButton(btn_text("pubmed", "PubMed"), callback_data="db_toggle_pubmed_entry"),
            InlineKeyboardButton(btn_text("libgen", "Libgen"), callback_data="db_toggle_libgen_entry")
        ],
        [InlineKeyboardButton("Продолжить ➡️", callback_data="db_done_entry")],
        [InlineKeyboardButton("« В начало (Отмена)", callback_data="cancel_conv")]
    ]
    await message_interface.reply_text(
        "Шаг 1: Выберите базы данных для поиска (можно несколько).\n"
        "Это важно, так как для разных баз могут потребоваться ключевые слова на разных языках.\n"
        "⚠️ *Внимание:* Поиск по Libgen может быть нестабилен или недоступен из-за ограничений доступа к сайту.\n"
        "Нажмите 'Продолжить', когда закончите.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )
    return STATE_DATABASES 

async def database_toggle_entry_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    db_id_toggle = query.data.split("_entry")[0].split("db_toggle_")[-1]

    current_selection = context.user_data.get("databases_selection", [])
    if db_id_toggle in current_selection:
        current_selection.remove(db_id_toggle)
    else:
        current_selection.append(db_id_toggle)
    context.user_data["databases_selection"] = current_selection

    def btn_text(db_id, text):
        return f"✅ {text}" if db_id in current_selection else text

    keyboard_updated = [
        [
            InlineKeyboardButton(btn_text("arxiv", "arXiv"), callback_data="db_toggle_arxiv_entry"),
            InlineKeyboardButton(btn_text("pubmed", "PubMed"), callback_data="db_toggle_pubmed_entry"),
            InlineKeyboardButton(btn_text("libgen", "Libgen"), callback_data="db_toggle_libgen_entry")
        ],
        [InlineKeyboardButton("Продолжить ➡️", callback_data="db_done_entry")],
        [InlineKeyboardButton("« В начало (Отмена)", callback_data="cancel_conv")]
    ]
    try:
        await query.edit_message_text(
            "Шаг 1: Выберите базы данных для поиска (можно несколько).\n"
            "Это важно, так как для разных баз могут потребоваться ключевые слова на разных языках.\n"
            "⚠️ *Внимание:* Поиск по Libgen может быть нестабилен или недоступен из-за ограничений доступа к сайту.\n"
            "Нажмите 'Продолжить', когда закончите.",
            reply_markup=InlineKeyboardMarkup(keyboard_updated),
            parse_mode="Markdown"
        )
    except Exception as e:
        logging.info(f"Error editing message for DB toggle (entry): {e}")
    return STATE_DATABASES

async def database_done_entry_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    selected_dbs = context.user_data.get("databases_selection", [])
    if not selected_dbs:
        await query.message.reply_text("Пожалуйста, выберите хотя бы одну базу данных или нажмите 'Отмена'.")
        return STATE_DATABASES 

    context.user_data["databases"] = "_".join(sorted(selected_dbs))
    
    lang_advice = ""
    if "arxiv" in selected_dbs or "pubmed" in selected_dbs:
        lang_advice += "Для arXiv и PubMed рекомендуется использовать ключевые слова на *английском языке* для наилучших результатов. "
    if "libgen" in selected_dbs:
        lang_advice += "Для Libgen можно использовать ключевые слова на *любом языке*."
    
    await query.message.reply_text(
        f"Шаг 2: Отправьте мне ключевые слова (слова или фразы, разделённые запятыми).\n{lang_advice.strip()}\n"
        "Это поможет точно сфокусировать поиск на интересующей вас теме.",
        parse_mode="Markdown"
    )
    return STATE_KEYWORDS

async def receive_keywords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["keywords"] = update.message.text.strip()
    keyboard = [
        [
            InlineKeyboardButton("AND (все слова)", callback_data="and"),
            InlineKeyboardButton("OR (хотя бы одно)", callback_data="or")
        ],
        [InlineKeyboardButton("« Назад (выбор баз)", callback_data="back_to_databases_entry_state")],
    ]
    await update.message.reply_text(
        "Шаг 3: Выберите режим поиска по ключевым словам.\n"
        "🔹 *AND*: Найдёт статьи, содержащие *все* указанные слова/фразы (сужает поиск).\n"
        "🔹 *OR*: Найдёт статьи, содержащие *хотя бы одно* из указанных слов/фраз (расширяет поиск).",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )
    return STATE_KEYWORD_MODE

async def back_to_databases_entry_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_databases_first_step(query.message, context)

async def keyword_mode_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    context.user_data["keyword_mode"] = query.data
    keyboard = [
        [InlineKeyboardButton("✍️ Ввести подробный запрос", callback_data="enter_query")],
        [InlineKeyboardButton("⏭️ Пропустить (использовать ключевые слова)", callback_data="skip_query")],
        [InlineKeyboardButton("« Назад (ключевые слова)", callback_data="back_to_keywords_state")],
    ]
    await query.message.reply_text(
        "Шаг 4: Хотите ввести более подробный запрос для ранжирования с помощью LLM?\n"
        "Это полезно, если вы ищете статьи по сложной или междисциплинарной теме. "
        "LLM сможет лучше понять ваш запрос и предложить наиболее релевантные результаты. "
        "Если пропустить, для ранжирования будут использованы только ключевые слова, которые вы ввели ранее.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_DETAILED_QUERY

async def back_to_keywords_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    
    selected_dbs = context.user_data.get("databases_selection", [])
    lang_advice = ""
    if "arxiv" in selected_dbs or "pubmed" in selected_dbs:
        lang_advice += "Для arXiv и PubMed рекомендуется использовать ключевые слова на *английском языке*. "
    if "libgen" in selected_dbs:
        lang_advice += "Для Libgen можно использовать ключевые слова на *любом языке*."

    await query.message.edit_text( 
        f"Хорошо, давайте вернемся к вводу ключевых слов.\n"
        f"Отправьте мне ключевые слова (слова или фразы, разделённые запятыми).\n{lang_advice.strip()}",
        reply_markup=None, 
        parse_mode="Markdown"
    )
    return STATE_KEYWORDS

async def detailed_query_router(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "enter_query":
        await query.message.reply_text(
            "Введите ваш подробный запрос для LLM (например, 'ищу статьи о влиянии изменения климата на биоразнообразие в арктических регионах, с фокусом на методах адаптации').\n"
            "Чем детальнее запрос, тем точнее LLM сможет подобрать статьи."
            )
        return STATE_DETAILED_QUERY 
    elif action == "skip_query":
        context.user_data["detailed_query"] = "" 
        return await ask_field_of_study(query.message, context)
    elif action == "back_to_keywords_state": 
        keyboard_km = [
            [
                InlineKeyboardButton("AND (все слова)", callback_data="and"),
                InlineKeyboardButton("OR (хотя бы одно)", callback_data="or")
            ],
            [InlineKeyboardButton("« Назад (выбор баз)", callback_data="back_to_databases_entry_state")],
        ]
        await query.message.edit_text( 
            "Шаг 3: Выберите режим поиска по ключевым словам.\n"
            "🔹 *AND*: Найдёт статьи, содержащие *все* указанные слова/фразы.\n"
            "🔹 *OR*: Найдёт статьи, содержащие *хотя бы одно* из указанных слов/фраз.",
            reply_markup=InlineKeyboardMarkup(keyboard_km),
            parse_mode="Markdown"
        )
        return STATE_KEYWORD_MODE
    return STATE_DETAILED_QUERY

async def receive_detailed_query(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    context.user_data["detailed_query"] = update.message.text.strip()
    return await ask_field_of_study(update.message, context)

async def ask_field_of_study(message_interface, context: ContextTypes.DEFAULT_TYPE): 
    keyboard = [
        [InlineKeyboardButton("🧬 Биология", callback_data="fos_Биология"), 
         InlineKeyboardButton("🧮 Математика", callback_data="fos_Математика")],
        [InlineKeyboardButton("💻 Комп. наука", callback_data="fos_Компьютерные науки"),
         InlineKeyboardButton("🔬 Физика", callback_data="fos_Физика")],
        [InlineKeyboardButton("🧪 Химия", callback_data="fos_Химия"), 
         InlineKeyboardButton("🌍 Науки о Земле", callback_data="fos_Науки о Земле")], 
        [InlineKeyboardButton("⏭️ Пропустить (общие науки)", callback_data="fos_skip_field")],
        [InlineKeyboardButton("✍️ Ввести свою", callback_data="fos_custom_field")],
        [InlineKeyboardButton("« Назад (запрос LLM)", callback_data="back_to_detailed_query_state")],
    ]
    await message_interface.reply_text(
        "Шаг 5: Выберите область науки для LLM или пропустите (будут 'общие науки').\n"
        "Указание области науки поможет LLM лучше понять контекст вашего запроса и точнее ранжировать статьи, "
        "особенно если ваши ключевые слова могут встречаться в разных дисциплинах.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_FIELD_OF_STUDY

async def back_to_detailed_query_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard_dq = [
        [InlineKeyboardButton("✍️ Ввести подробный запрос", callback_data="enter_query")],
        [InlineKeyboardButton("⏭️ Пропустить (использовать ключевые слова)", callback_data="skip_query")],
        [InlineKeyboardButton("« Назад (ключевые слова)", callback_data="back_to_keywords_state")],
    ]
    await query.message.edit_text(
        "Шаг 4: Хотите ввести более подробный запрос для ранжирования с помощью LLM?\n"
        "Это полезно, если вы ищете статьи по сложной или междисциплинарной теме. "
        "Если пропустить, для ранжирования будут использованы только ключевые слова.",
        reply_markup=InlineKeyboardMarkup(keyboard_dq)
    )
    return STATE_DETAILED_QUERY

async def field_of_study_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    choice = query.data 

    if choice == "fos_skip_field":
        context.user_data["field_of_study"] = "общие науки"
        return await ask_max_results(query.message, context)
    elif choice == "fos_custom_field":
        await query.message.reply_text("Введите вашу область науки (например, 'Социология', 'История искусств'):")
        return STATE_FIELD_OF_STUDY 
    elif choice.startswith("fos_"):
        context.user_data["field_of_study"] = choice.split("fos_")[1]
        return await ask_max_results(query.message, context)
    
    return STATE_FIELD_OF_STUDY

async def receive_field_of_study_text(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    context.user_data["field_of_study"] = update.message.text.strip()
    return await ask_max_results(update.message, context)

async def ask_max_results(message_interface, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("10", callback_data="maxres_10"), InlineKeyboardButton("20", callback_data="maxres_20")],
        [InlineKeyboardButton("30", callback_data="maxres_30"), InlineKeyboardButton("50", callback_data="maxres_50")], 
        [InlineKeyboardButton("✍️ Ввести своё", callback_data="maxres_custom")],
        [InlineKeyboardButton("⏭️ Пропустить (по умолч. 20)", callback_data="maxres_skip")],
        [InlineKeyboardButton("« Назад (область науки)", callback_data="back_to_field_state")],
    ]
    await message_interface.reply_text(
        "Шаг 6: Выберите максимальное количество статей для первоначального поиска (до LLM-ранжирования).\n"
        "Большее число может дать более широкий охват, но увеличит время обработки.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_MAX_RESULTS

async def back_to_field_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_field_of_study(query.message, context)

async def max_results_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    choice = query.data 

    if choice == "maxres_skip":
        context.user_data["max_results"] = 20 
        return await ask_top_count(query.message, context)
    elif choice == "maxres_custom":
        await query.message.reply_text("Введите желаемое число статей для поиска (например, 25, макс. 100):")
        return STATE_MAX_RESULTS_CUSTOM 
    elif choice.startswith("maxres_"):
        try:
            context.user_data["max_results"] = int(choice.split("maxres_")[1])
            return await ask_top_count(query.message, context)
        except ValueError:
            await query.message.reply_text("Неверное значение. Попробуйте снова.")
            return STATE_MAX_RESULTS 
    return STATE_MAX_RESULTS

async def receive_custom_max_results_text(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    text = update.message.text.strip()
    try:
        val = int(text)
        if val < 1: val = 1
        if val > 100: val = 100
        context.user_data["max_results"] = val
        return await ask_top_count(update.message, context)
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число. Например, 15.")
        return STATE_MAX_RESULTS_CUSTOM

async def ask_top_count(message_interface, context: ContextTypes.DEFAULT_TYPE):
    max_r = context.user_data.get("max_results", 20)
    
    options = [1, 3, 5, 10] 
    valid_options = [opt for opt in options if opt <= max_r]
    
    keyboard_buttons_row = []
    if valid_options:
         keyboard_buttons_row = [InlineKeyboardButton(str(opt), callback_data=f"top_{opt}") for opt in valid_options]
    
    keyboard = [keyboard_buttons_row] if keyboard_buttons_row else []

    if max_r not in valid_options and max_r > 0: 
        keyboard.append([InlineKeyboardButton(f"Показать все ({max_r})", callback_data=f"top_{max_r}")])
    
    keyboard.append([InlineKeyboardButton("« Назад (макс. статей)", callback_data="back_to_max_results_state")])

    await message_interface.reply_text(
        f"Шаг 7: Сколько лучших статей показать после LLM-ранжирования (из найденных ~{max_r})?\n"
        "Это количество статей, которое вы увидите в итоге.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_TOP_COUNT

async def back_to_max_results_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_max_results(query.message, context)

async def top_count_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    choice = query.data 

    if choice.startswith("top_"):
        try:
            top_val = int(choice.split("top_")[1])
            max_r = context.user_data.get("max_results", 20)
            context.user_data["top_count"] = min(top_val, max_r)
            return await ask_days(query.message, context)
        except ValueError:
            await query.message.reply_text("Неверное значение. Попробуйте снова.")
            return await ask_top_count(query.message, context) 
    return STATE_TOP_COUNT

async def ask_days(message_interface, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("За 1 день", callback_data="days_1"), InlineKeyboardButton("За 7 дней", callback_data="days_7")],
        [InlineKeyboardButton("За 30 дней", callback_data="days_30"), InlineKeyboardButton("Без ограничений по дате", callback_data="days_0")],
        [InlineKeyboardButton("✍️ Ввести своё (дней)", callback_data="days_custom")],
        [InlineKeyboardButton("« Назад (топ статей)", callback_data="back_to_top_count_state")],
    ]
    await message_interface.reply_text(
        "Шаг 8: Укажите глубину поиска в днях (относительно текущей даты) или выберите 'без ограничений'.\n"
        "Это поможет найти самые свежие публикации или, наоборот, провести более широкий исторический поиск.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_DAYS

async def back_to_top_count_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_top_count(query.message, context)

async def days_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    choice = query.data 

    if choice == "days_custom":
        await query.message.reply_text("Введите число дней для поиска (например, 14, 0 для без ограничений):")
        return STATE_DAYS
    elif choice.startswith("days_"):
        try:
            context.user_data["days"] = int(choice.split("days_")[1])
            return await ask_min_citations(query.message, context)
        except ValueError: 
            await query.message.reply_text("Ошибка. Попробуйте снова.")
            return STATE_DAYS
    return STATE_DAYS

async def receive_custom_days_text(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    text = update.message.text.strip()
    try:
        days_val = int(text)
        if days_val < 0: days_val = 0 
        context.user_data["days"] = days_val
        return await ask_min_citations(update.message, context)
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число дней (0 для без ограничений).")
        return STATE_DAYS

async def ask_min_citations(message_interface, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Любое", callback_data="cite_0"), InlineKeyboardButton("> 5", callback_data="cite_5")],
        [InlineKeyboardButton("> 10", callback_data="cite_10"), InlineKeyboardButton("> 20", callback_data="cite_20")],
        [InlineKeyboardButton("✍️ Ввести своё", callback_data="cite_custom")],
        [InlineKeyboardButton("« Назад (дни поиска)", callback_data="back_to_days_state")],
    ]
    await message_interface.reply_text(
        "Шаг 9: Укажите минимальное число цитирований для статей (кроме Libgen).\n"
        "Это может помочь отфильтровать менее влиятельные работы. "
        "Обратите внимание: для книг из Libgen этот фильтр не применяется, так как информация о цитированиях для них обычно отсутствует.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_MIN_CITATIONS

async def back_to_days_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_days(query.message, context)

async def min_citations_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    choice = query.data 

    if choice == "cite_custom":
        await query.message.reply_text("Введите минимальное число цитирований (например, 15, 0 для любого):")
        return STATE_MIN_CITATIONS 
    elif choice.startswith("cite_"):
        try:
            context.user_data["min_citations"] = int(choice.split("cite_")[1])
            return await ask_reviews_only(query.message, context)
        except ValueError:
            await query.message.reply_text("Ошибка. Попробуйте снова.")
            return STATE_MIN_CITATIONS
    return STATE_MIN_CITATIONS

async def receive_custom_citations_text(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    text = update.message.text.strip()
    try:
        cites_val = int(text)
        if cites_val < 0: cites_val = 0
        context.user_data["min_citations"] = cites_val
        return await ask_reviews_only(update.message, context) 
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число (0 для любого количества).")
        return STATE_MIN_CITATIONS 

async def ask_reviews_only(message_interface, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔍 Все статьи", callback_data="rev_all"), 
         InlineKeyboardButton("🧐 Только обзоры (review/survey)", callback_data="rev_reviews")],
        [InlineKeyboardButton("« Назад (цитирования)", callback_data="back_to_citations_state")],
    ]
    await message_interface.reply_text(
        "Шаг 10: Искать только обзорные статьи (review/survey papers) или все подряд?\n"
        "Обзорные статьи обобщают существующие исследования по теме и могут быть хорошей отправной точкой.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_REVIEWS_ONLY

async def back_to_citations_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_min_citations(query.message, context)

async def reviews_only_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    choice = query.data 
    context.user_data["review_only"] = (choice == "rev_reviews")
    return await ask_periodic_updates(query.message, context)

async def ask_periodic_updates(message_interface, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Раз в 1 день", callback_data="period_1"), 
         InlineKeyboardButton("Раз в 7 дней", callback_data="period_7")],
        [InlineKeyboardButton("Раз в 30 дней", callback_data="period_30"), 
         InlineKeyboardButton("Не подписываться", callback_data="period_0")],
        [InlineKeyboardButton("« Назад (тип обзоров)", callback_data="back_to_reviews_state")],
    ]
    await message_interface.reply_text(
        "Шаг 11: Хотите подписаться на периодическую рассылку с новыми статьями по этому запросу? 📬\n"
        "Если вы выберете период, бот будет автоматически проверять наличие новых публикаций и присылать их вам.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_PERIODIC_UPDATES

async def back_to_reviews_state_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_reviews_only(query.message, context)

async def periodic_updates_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    choice = query.data 

    if choice.startswith("period_"):
        try:
            period = int(choice.split("period_")[1])
            context.user_data["subscription_period"] = period
            if period <= 0:
                context.user_data["subscription_hour"] = 0
                context.user_data["subscription_minute"] = 0
                return await confirm_params(query.message, context)
            else:
                return await ask_subscription_hour(query.message, context)
        except ValueError:
            await query.message.reply_text("Ошибка. Попробуйте снова.")
            return STATE_PERIODIC_UPDATES
    return STATE_PERIODIC_UPDATES

async def ask_subscription_hour(message_interface, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []    

    for row in range(6):
        button_row = []
        for col in range(4):
            hour = row * 4 + col
            if hour < 24:
                button_row.append(InlineKeyboardButton(f"{hour:02d}", callback_data=f"hour_{hour}"))
        if button_row:
            keyboard.append(button_row)
    
    keyboard.append([InlineKeyboardButton("« Назад (период)", callback_data="back_to_periodic_updates")])
    
    await message_interface.reply_text(
        "🕐 Выберите час для рассылки (время UTC):\n\n"
        "💡 Примеры времени UTC:\n"
        "• 06:00 UTC = 09:00 MSK\n"
        "• 12:00 UTC = 15:00 MSK\n"
        "• 18:00 UTC = 21:00 MSK",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_SUBSCRIPTION_HOUR


async def back_to_periodic_updates_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_periodic_updates(query.message, context)

async def subscription_hour_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("hour_"):
        try:
            hour = int(query.data.split("hour_")[1])
            context.user_data["subscription_hour"] = hour
            return await ask_subscription_minute(query.message, context)
        except ValueError:
            await query.message.reply_text("Ошибка выбора часа. Попробуйте снова.")
            return await ask_subscription_hour(query.message, context)
    
    return STATE_SUBSCRIPTION_HOUR

async def ask_subscription_minute(message_interface, context: ContextTypes.DEFAULT_TYPE):
    hour = context.user_data.get("subscription_hour", 0)    

    popular_minutes = [0, 15, 30, 45]
    
    keyboard = []    

    popular_row = []
    for minute in popular_minutes:
        popular_row.append(InlineKeyboardButton(f"{hour:02d}:{minute:02d}", callback_data=f"minute_{minute}"))
    keyboard.append(popular_row)    

    additional_rows = []
    other_minutes = [5, 10, 20, 25, 35, 40, 50, 55]
    
    for i in range(0, len(other_minutes), 4):
        row = []
        for j in range(4):
            if i + j < len(other_minutes):
                minute = other_minutes[i + j]
                row.append(InlineKeyboardButton(f":{minute:02d}", callback_data=f"minute_{minute}"))
        if row:
            keyboard.append(row)    

    keyboard.append([InlineKeyboardButton("✍️ Ввести другое время", callback_data="minute_custom")])
    keyboard.append([InlineKeyboardButton("« Назад (час)", callback_data="back_to_hour_selection")])
    
    await message_interface.reply_text(
        f"⏰ Выберите минуты для рассылки в {hour:02d}:XX (время UTC):\n\n"
        f"Популярные варианты в первом ряду, дополнительные ниже.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATE_SUBSCRIPTION_MINUTE


async def back_to_hour_selection_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    return await ask_subscription_hour(query.message, context)

async def subscription_minute_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "minute_custom":
        hour = context.user_data.get("subscription_hour", 0)
        await query.message.reply_text(
            f"✍️ Введите минуты для времени рассылки {hour:02d}:XX\n"
            f"Укажите число от 0 до 59 (например: 7, 23, 45)"
        )
        return STATE_SUBSCRIPTION_MINUTE_CUSTOM
    elif query.data.startswith("minute_"):
        try:
            minute = int(query.data.split("minute_")[1])
            if 0 <= minute <= 59:
                context.user_data["subscription_minute"] = minute
                return await confirm_params(query.message, context)
            else:
                await query.message.reply_text("Ошибка: минуты должны быть от 0 до 59. Попробуйте снова.")
                return await ask_subscription_minute(query.message, context)
        except ValueError:
            await query.message.reply_text("Ошибка выбора минуты. Попробуйте снова.")
            return await ask_subscription_minute(query.message, context)
    
    return STATE_SUBSCRIPTION_MINUTE


STATE_SUBSCRIPTION_MINUTE_CUSTOM = 16

async def receive_custom_subscription_minute(update: Update, context: ContextTypes.DEFAULT_TYPE):

    text = update.message.text.strip()
    try:
        minute = int(text)
        if 0 <= minute <= 59:
            context.user_data["subscription_minute"] = minute
            hour = context.user_data.get("subscription_hour", 0)
            await update.message.reply_text(
                f"✅ Время рассылки установлено: {hour:02d}:{minute:02d} UTC"
            )
            return await confirm_params(update.message, context)
        else:
            await update.message.reply_text(
                "❌ Минуты должны быть от 0 до 59. Пожалуйста, введите корректное значение:"
            )
            return STATE_SUBSCRIPTION_MINUTE_CUSTOM
    except ValueError:
        await update.message.reply_text(
            "❌ Пожалуйста, введите число от 0 до 59:"
        )
        return STATE_SUBSCRIPTION_MINUTE_CUSTOM



def should_send_subscription_now(last_update_val: Union[str, date, datetime], 
                                period_days: int,
                                subscription_hour: int = 0,
                                subscription_minute: int = 0) -> bool:

    if period_days <= 0:
        return False    

    last_update_dt: datetime
    if isinstance(last_update_val, str):
        last_update_dt = parse_datetime_from_string(last_update_val) 
    elif isinstance(last_update_val, date): 
        last_update_dt = datetime.combine(last_update_val, datetime.min.time())
    elif isinstance(last_update_val, datetime): 
        last_update_dt = last_update_val
    else: 
        last_update_dt = datetime(1970, 1, 1)

    now_utc = datetime.now(timezone.utc)
    last_update_date = last_update_dt.date()
    today_utc = now_utc.date()    

    days_passed = (today_utc - last_update_date).days
    
    if days_passed < period_days:
        return False    

    scheduled_time_today = datetime.combine(
        today_utc, 
        datetime.min.time().replace(hour=subscription_hour, minute=subscription_minute)
    ).replace(tzinfo=timezone.utc)    

    return now_utc >= scheduled_time_today



async def confirm_params(message_interface, context: ContextTypes.DEFAULT_TYPE):
    data = context.user_data
    db_list = data.get('databases_selection', []) 
    if not db_list and data.get('databases'): 
        db_list = data.get('databases').split('_')
    db_display = ", ".join(db_list) if db_list else "не выбраны"
    
    period = data.get('subscription_period', 0)
    time_display = ""
    if period > 0:
        hour = data.get('subscription_hour', 0)
        minute = data.get('subscription_minute', 0)
        time_display = f" в {hour:02d}:{minute:02d} (UTC)"
    
    summary = (
        f"🔍 *Проверьте параметры поиска:*\n\n"
        f"📚 Базы данных: `{db_display}`\n"
        f"🔑 Ключевые слова: `{data.get('keywords', 'не заданы')}`\n"
        f"⚖️ Режим слов: `{data.get('keyword_mode', 'or').upper()}`\n"
        f"🧠 Запрос для LLM: `{data.get('detailed_query') or 'как ключевые слова'}`\n"
        f"🔬 Область науки: `{data.get('field_of_study', 'общие науки')}`\n"
        f"🔢 Искать статей (до LLM): `{data.get('max_results', 20)}`\n"
        f"🏆 Показать топ статей: `{data.get('top_count', 3)}`\n"
        f"📅 Глубина поиска: `за последние {data.get('days', 0)} дней` (0 – без ограничений)\n"
        f"📊 Мин. цитирований: `{data.get('min_citations', 0)}` (0 – без ограничений, кроме Libgen)\n"
        f"🧐 Только обзоры: `{'Да' if data.get('review_only', False) else 'Нет'}`\n"
        f"📬 Период подписки: `{period} дней` (0 – без подписки){time_display}\n\n"
        "Всё верно?"
    )
    keyboard = [
        [InlineKeyboardButton("✅ Да, запустить поиск!", callback_data="run_search_final")],
        [InlineKeyboardButton("« Отмена (в начало)", callback_data="cancel_conv_final")],
    ]
    await message_interface.reply_text(summary, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))
    return STATE_CONFIRM_PARAMS

async def confirm_params_chosen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "run_search_final":
        await query.edit_message_text("Выполняю поиск и ранжирование... Это может занять некоторое время. Пожалуйста, подождите. ⏳", reply_markup=None)

        async def send_progress(msg_text): 
            try:
                await query.message.reply_text(f"⚙️ {msg_text}")
            except Exception as e:
                logging.info(f"Could not send progress message: {e}")

        period = context.user_data.get("subscription_period", 0)
        if period > 0:
            db_string_for_storage = context.user_data.get("databases")
            if not db_string_for_storage and "databases_selection" in context.user_data:
                 db_string_for_storage = "_".join(sorted(context.user_data.get("databases_selection", [])))

            now_utc = datetime.now(timezone.utc)
            sub_hour = context.user_data.get("subscription_hour", 0)
            sub_minute = context.user_data.get("subscription_minute", 0)

            if now_utc.hour > sub_hour or (now_utc.hour == sub_hour and now_utc.minute >= sub_minute):
                initial_last_update = now_utc.date()
                logging.info(f"Подписка создана после времени рассылки. last_update = {initial_last_update}")
            else:
                initial_last_update = now_utc.date() - timedelta(days=period)
                logging.info(f"Подписка создана до времени рассылки. last_update = {initial_last_update}")

            sub_data_for_db = {
                "keywords": context.user_data.get("keywords", ""),
                "field_of_study": context.user_data.get("field_of_study", "общие науки"),
                "subscription_period": period,
                "last_update": initial_last_update,
                "max_results": context.user_data.get("max_results", 20),
                "days": context.user_data.get("days", 0), 
                "review_only": context.user_data.get("review_only", False),
                "keyword_mode": context.user_data.get("keyword_mode", "or"),
                "detailed_query": context.user_data.get("detailed_query", ""),
                "min_citations": context.user_data.get("min_citations", 0),
                "databases": db_string_for_storage or "arxiv_pubmed_libgen",
                "subscription_hour": sub_hour,
                "subscription_minute": sub_minute,
            }
            try:
                await update_subscription_db(update.effective_chat.id, sub_data_for_db)
                await query.message.reply_text(f"Вы успешно подписались на обновления каждые {period} дней по этому запросу. ✅\n"
                                               f"Рассылка будет приходить примерно в {sub_hour:02d}:{sub_minute:02d} UTC.")
            except Exception as e:
                logging.error(f"Failed to save subscription for chat {update.effective_chat.id}: {e}", exc_info=True)
                await query.message.reply_text("Не удалось сохранить настройки подписки. Поиск будет выполнен без подписки. 😥")
        
        if "databases_selection" in context.user_data and not context.user_data.get("databases"):
            context.user_data["databases"] = "_".join(sorted(context.user_data["databases_selection"]))

        papers = await perform_search_and_ranking(context.user_data, 
                                                  progress_callback=send_progress,
                                                  date_from=None) 
        if not papers:
            await query.message.reply_text("По вашему запросу ничего не найдено. 😔 Попробуйте изменить параметры поиска или ключевые слова.")
            context.user_data.clear()
            return ConversationHandler.END

        top_count_to_show = context.user_data.get("top_count", 3)
        top_papers_list = papers[:min(top_count_to_show, len(papers))]

        await query.message.reply_text(f"Готово! 🎉 Найдено статей: {len(papers)}. Показываю топ-{len(top_papers_list)}:")
        
        context.user_data["search_results_full"] = papers 

        for displayed_idx, paper_to_display in enumerate(top_papers_list): 
            authors_text = paper_to_display.get("authors") or "Авторы не указаны"
            title = paper_to_display.get("title", "Без названия")
            date_p_str = paper_to_display.get("publication_date", "N/A")
            cites = paper_to_display.get("cited_by", "N/A")
            rating = paper_to_display.get("rating", 0)
            doi_or_id = paper_to_display.get("doi", "") or paper_to_display.get("paperId", "")
            source = paper_to_display.get("source", "Неизвестный источник")

            link = "N/A"
            
            if source == "arXiv":
                paper_id_val = paper_to_display.get("paperId") or doi_or_id.replace("arxiv:", "")
                link = f"https://arxiv.org/abs/{paper_id_val}"
            elif source == "PMC":
                pmcid = paper_to_display.get("paperId") 
                if pmcid:
                    if not pmcid.startswith("PMC"): pmcid = "PMC" + pmcid.replace("PMC", "")
                    link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}"
                else: 
                    link = f"https://www.ncbi.nlm.nih.gov/pmc/?term={requests.utils.quote(title)}"
            elif source == "Libgen":
                quoted_title = requests.utils.quote(paper_to_display.get('title', ''))
                dl_links = paper_to_display.get("download_links", {})
                if isinstance(dl_links, dict):
                    link = dl_links.get("direct") or dl_links.get("mirror_1") or f"https://libgen.is/search.php?req={quoted_title}"
                else:
                    link = f"https://libgen.is/search.php?req={quoted_title}"

            text_html = (
                f"<b>{displayed_idx + 1}. {title}</b>\n" 
                f"👤 Автор(ы): {authors_text}\n"
                f"📅 Дата: {date_p_str}\n"
                f"📊 Цитирований: {cites if cites is not None else 'N/A'}\n"
                f"⭐ Рейтинг (LLM): {rating}\n"
                f"🆔 DOI/ID: {doi_or_id}\n"
                f"📚 Источник: {source}\n"
                f"🔗 Ссылка: <a href='{link}'>{link}</a>" 
            )
            
            callback_idx = displayed_idx
            keyboard_abstract = [[InlineKeyboardButton("📖 Показать аннотацию", callback_data=f"show_abstract_{callback_idx}")]]
            reply_markup_abstract = InlineKeyboardMarkup(keyboard_abstract)
            
            await query.message.reply_text(
                text_html,
                parse_mode="HTML",
                disable_web_page_preview=True,
                reply_markup=reply_markup_abstract
            )
            await asyncio.sleep(0.3) 

        keyboard_after_results = [
            [InlineKeyboardButton("📥 Скачать все результаты (JSON)", callback_data="download_results_action")],
            [InlineKeyboardButton("🔄 Новый поиск", callback_data="start_search")], 
            [InlineKeyboardButton("🚪 Завершить", callback_data="cancel_direct")] 
        ]
        await query.message.reply_text(
            "Поиск завершен. Что дальше?",
            reply_markup=InlineKeyboardMarkup(keyboard_after_results)
        )
        return ConversationHandler.END 

    elif action == "cancel_conv_final": 
        await query.message.reply_text("Поиск отменен. Начните заново с /search или кнопки 'Поиск статей'.")
        context.user_data.clear()
        return ConversationHandler.END
    
    return STATE_CONFIRM_PARAMS

async def show_abstract_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    try:
        paper_index_in_full_list = int(query.data.split('_')[-1])
    except (IndexError, ValueError):
        await query.message.reply_text("Ошибка: неверный формат запроса аннотации.")
        return

    full_papers_list = context.user_data.get("search_results_full")
    
    if not full_papers_list or not (0 <= paper_index_in_full_list < len(full_papers_list)):
        await query.message.reply_text("Ошибка: информация об аннотации не найдена или устарела. Пожалуйста, начните новый поиск.")
        return
    
    abstract_text = full_papers_list[paper_index_in_full_list].get("abstract", "Аннотация отсутствует.")
    
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception as e:
        logging.info(f"Could not remove keyboard from abstract message: {e}")

    max_length = 4096
    full_message = f"📜 <b>Аннотация:</b>\n<blockquote>{abstract_text}</blockquote>"
    
    if len(full_message) <= max_length:
        await query.message.reply_text(
            full_message, 
            parse_mode="HTML", 
            disable_web_page_preview=True
        )
    else:
        max_chunk_text_len = max_length - 100 
        
        chunks = []
        remaining_text = abstract_text
        while len(remaining_text) > 0:
            chunks.append(remaining_text[:max_chunk_text_len])
            remaining_text = remaining_text[max_chunk_text_len:]

        for i, chunk in enumerate(chunks):
            if i == 0:
                header = f"📜 <b>Аннотация (часть {i+1}/{len(chunks)}):</b>\n"
            else:
                header = f"<b>(продолжение, часть {i+1}/{len(chunks)})</b>\n"
            
            message_chunk = f"{header}<blockquote>{chunk}</blockquote>"
            
            await query.message.reply_text(
                message_chunk,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            await asyncio.sleep(0.3)

async def download_results_action_callback(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()

    full_results = context.user_data.get("search_results_full")
    if not full_results:
        await query.message.reply_text("Нет результатов для скачивания или данные устарели. Пожалуйста, начните новый поиск.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"papers_search_results_{timestamp}.json" 
    
    save_papers(full_results, filename) 

    file_sent_successfully = False
    try:
        with open(filename, "rb") as f:
            await query.message.reply_document(
                document=InputFile(f, filename=filename), 
                caption=f"Файл со всеми ({len(full_results)}) найденными результатами ({filename})."
            )
        file_sent_successfully = True
    except Exception as e:
        logging.error(f"Ошибка отправки файла: {e}", exc_info=True)
        await query.message.reply_text("Не удалось отправить файл. 😔")
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as e_remove:
                logging.error(f"Could not remove temporary file {filename}: {e_remove}")
    
    message_after_download = "Файл результатов был отправлен." if file_sent_successfully else "Была попытка отправки файла результатов."
    
    keyboard_after_download = [
        [InlineKeyboardButton("🔄 Новый поиск", callback_data="start_search")],
        [InlineKeyboardButton("🚪 Завершить", callback_data="cancel_direct")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard_after_download)
    
    await query.message.reply_text(
        f"{message_after_download} Что вы хотите сделать дальше?",
        reply_markup=reply_markup
    )
    
async def cancel_direct_callback(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("Действие завершено. Спасибо! 😊", reply_markup=None)
    context.user_data.clear()

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    
    chat_id = None
    if isinstance(update, Update) and update.effective_chat:
        chat_id = update.effective_chat.id
    
    from telegram.error import NetworkError, BadRequest, TimedOut, RetryAfter
    if isinstance(context.error, (NetworkError, TimedOut)):
        error_message = "Возникла сетевая ошибка или истекло время ожидания. Пожалуйста, попробуйте ваш запрос позже. 🔌⏳"
    elif isinstance(context.error, BadRequest):
        if "Message is not modified" in str(context.error):
            return
        error_message = f"Произошла ошибка с запросом: {context.error}. Пожалуйста, проверьте введенные данные."
    elif isinstance(context.error, RetryAfter):
        error_message = f"Слишком много запросов. Пожалуйста, подождите {context.error.retry_after} секунд и попробуйте снова."
    else:
        error_message = "Произошла непредвиденная ошибка. 😥 Мы уже работаем над этим. Попробуйте позже."

    if chat_id:
        try:
            await context.bot.send_message(chat_id=chat_id, text=error_message)
        except Exception as e_send:
            logging.error(f"Failed to send error message to user {chat_id}: {e_send}")

async def main():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        print("Ошибка: не указан TELEGRAM_TOKEN в переменных окружения.")
        return

    await init_db()

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("search", start_search_command_entry), 
            CallbackQueryHandler(start_search_trigger, pattern="^start_search$"), 
        ],
        states={
            STATE_DATABASES: [ 
                CallbackQueryHandler(database_toggle_entry_handler, pattern="^db_toggle_(arxiv|pubmed|libgen)_entry$"),
                CallbackQueryHandler(database_done_entry_handler, pattern="^db_done_entry$"),
                CallbackQueryHandler(cancel_trigger, pattern="^cancel_conv$") 
            ],
            STATE_KEYWORDS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_keywords),
                CallbackQueryHandler(back_to_databases_entry_state_handler, pattern="^back_to_databases_entry_state$")
            ],
            STATE_KEYWORD_MODE: [
                CallbackQueryHandler(keyword_mode_chosen, pattern="^(and|or)$"), 
                CallbackQueryHandler(back_to_keywords_state_handler, pattern="^back_to_keywords_state$") 
            ],
            STATE_DETAILED_QUERY: [
                CallbackQueryHandler(detailed_query_router, pattern="^(enter_query|skip_query|back_to_keywords_state)$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_detailed_query) 
            ],
            STATE_FIELD_OF_STUDY: [
                CallbackQueryHandler(back_to_detailed_query_state_handler, pattern="^back_to_detailed_query_state$"),
                CallbackQueryHandler(field_of_study_chosen_handler, pattern="^fos_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_field_of_study_text),
            ],
            STATE_MAX_RESULTS: [
                CallbackQueryHandler(back_to_field_state_handler, pattern="^back_to_field_state$"),
                CallbackQueryHandler(max_results_chosen_handler, pattern="^maxres_"),
            ],
            STATE_MAX_RESULTS_CUSTOM: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_custom_max_results_text)],
            STATE_TOP_COUNT: [
                CallbackQueryHandler(back_to_max_results_state_handler, pattern="^back_to_max_results_state$"),
                CallbackQueryHandler(top_count_chosen_handler, pattern="^top_"),
            ],
            STATE_DAYS: [
                CallbackQueryHandler(back_to_top_count_state_handler, pattern="^back_to_top_count_state$"),
                CallbackQueryHandler(days_chosen_handler, pattern="^days_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_custom_days_text), 
            ],
            STATE_MIN_CITATIONS: [
                CallbackQueryHandler(back_to_days_state_handler, pattern="^back_to_days_state$"),
                CallbackQueryHandler(min_citations_chosen_handler, pattern="^cite_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_custom_citations_text), 
            ],
            STATE_REVIEWS_ONLY: [
                CallbackQueryHandler(back_to_citations_state_handler, pattern="^back_to_citations_state$"), 
                CallbackQueryHandler(reviews_only_chosen_handler, pattern="^rev_"),
            ],
            STATE_PERIODIC_UPDATES: [
                CallbackQueryHandler(back_to_reviews_state_handler, pattern="^back_to_reviews_state$"),
                CallbackQueryHandler(periodic_updates_chosen_handler, pattern="^period_"),
            ],
            STATE_SUBSCRIPTION_HOUR: [
                CallbackQueryHandler(back_to_periodic_updates_handler, pattern="^back_to_periodic_updates$"),
                CallbackQueryHandler(subscription_hour_chosen_handler, pattern="^hour_"),
            ],
            
            STATE_SUBSCRIPTION_MINUTE: [
                CallbackQueryHandler(back_to_hour_selection_handler, pattern="^back_to_hour_selection$"),
                CallbackQueryHandler(subscription_minute_chosen_handler, pattern="^minute_"),
            ],
            STATE_SUBSCRIPTION_MINUTE_CUSTOM: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_custom_subscription_minute)
            ],

            STATE_CONFIRM_PARAMS: [
                CallbackQueryHandler(confirm_params_chosen_handler, pattern="^(run_search_final|cancel_conv_final)$"),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_trigger), 
            CallbackQueryHandler(cancel_trigger, pattern="^cancel$"), 
            CallbackQueryHandler(cancel_trigger, pattern="^cancel_conv$"), 
            CallbackQueryHandler(cancel_trigger, pattern="^cancel_conv_final$"), 
        ],
        per_message=False, 
        allow_reentry=True 
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(conv_handler) 

    application.add_handler(CallbackQueryHandler(show_abstract_callback, pattern="^show_abstract_"))
    application.add_handler(CallbackQueryHandler(unsubscribe_subscription, pattern="^unsubscribe$"))
    application.add_handler(CallbackQueryHandler(download_results_action_callback, pattern="^download_results_action$"))
    application.add_handler(CallbackQueryHandler(cancel_direct_callback, pattern="^cancel_direct$")) 

    application.add_error_handler(error_handler)

    if SCHEDULER_AVAILABLE: 
        scheduler = AsyncIOScheduler(timezone=utc)
        scheduler.add_job(
            check_subscriptions, 
            "interval", 
            minutes=10,
            args=[application]
        )
        scheduler.start()
        logging.info("Планировщик задач APScheduler запущен (интервал: 10 мин, часовой пояс: UTC).")
    else:
        logging.warning("Планировщик APScheduler не запущен. Установите: pip install apscheduler pytz")

    print("Бот запущен. Ожидаю команды /start или /search...")

    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
