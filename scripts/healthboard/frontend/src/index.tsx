// SPDX-FileCopyrightText: 2026 top-papers-graph contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import 'antd/dist/reset.css';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
