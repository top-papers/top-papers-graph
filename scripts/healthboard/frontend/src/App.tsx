import React from 'react';
import { Layout } from 'antd';
import SystemInfo from './components/SystemInfo';
import './App.css';

const { Header, Content } = Layout;

function App(): JSX.Element {
  return (
    <Layout className="app">
      <Header className="header">
        <h1 style={{ color: 'white', margin: 0 }}>System Monitor Dashboard</h1>
      </Header>
      <Content style={{ padding: '20px' }}>
        <SystemInfo />
      </Content>
    </Layout>
  );
}

export default App;
