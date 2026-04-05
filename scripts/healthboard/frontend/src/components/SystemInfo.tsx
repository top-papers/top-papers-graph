import React, { useCallback, useMemo, useState } from 'react';
import { Card, Row, Col, Statistic, Spin, Alert, Button, Space, Tag, Table } from 'antd';
import { ReloadOutlined, PlayCircleOutlined, StopOutlined, FolderAddOutlined, PlusOutlined, MinusOutlined } from '@ant-design/icons';
import useSWR from 'swr';
import useSWRMutation from 'swr/mutation';
import { ApiResponse, RamData, DiskData, DockerStatus, UseSystemDataReturn, ApiData, CliOutput } from '../types/system';

// Fetcher function with TypeScript typing
async function fetcher<T extends ApiData>(url: string): Promise<ApiResponse<T>> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};

const useSystemData = (): UseSystemDataReturn => {
  // Use SWR for all endpoints
  const {
    data: ramData,
    error: ramError,
    mutate: mutateRam
  } = useSWR<ApiResponse<RamData>>('/api/system/ram', fetcher, {
    refreshInterval: 8000,
    revalidateOnFocus: false,
    dedupingInterval: 4000
  });

  const {
    data: diskData,
    error: diskError,
    mutate: mutateDisk
  } = useSWR<ApiResponse<DiskData>>('/api/system/disk', fetcher, {
    refreshInterval: 8000,
    revalidateOnFocus: false,
    dedupingInterval: 4000
  });

  const {
    data: dockerData,
    error: dockerError,
    mutate: mutateDocker
  } = useSWR<ApiResponse<DockerStatus>>('/api/docker/status', fetcher, {
    refreshInterval: 10000, // Docker status updates less frequently
    revalidateOnFocus: false,
    dedupingInterval: 5000
  });

  // Mutation hooks for Docker actions
  const { trigger: triggerStartDocker } = useSWRMutation(
    '/api/docker/start',
    async (url: string) => {
      const response = await fetch(url, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to start Docker Compose. ' + response.statusText);
      const data: ApiResponse<CliOutput> = await response.json();
      if (data.status !== "success") {
        throw Error(data.message ?? "Failed to start containers, no details");
      }
      return data;
    }
  );

  const { trigger: triggerStopDocker } = useSWRMutation(
    '/api/docker/stop',
    async (url: string) => {
      const response = await fetch(url, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to stop containers. ' + response.statusText);
      const data: ApiResponse<CliOutput> = await response.json();
      if (data.status !== "success") {
        throw Error(data.message ?? "Failed to stop containers, no details");
      }
      return data;
    }
  );

  const { trigger: triggerCreateContainers } = useSWRMutation(
    '/api/docker/create-containers',
    async (url) => {
      const response = await fetch(url, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to create containers. ' + response.statusText);
      const data: ApiResponse<CliOutput> = await response.json();
      if (data.status !== "success") {
        throw Error(data.message ?? "Failed to create containers, no details");
      }
      return data;
    }
  );

  const { trigger: triggerRemoveContainers } = useSWRMutation(
    '/api/docker/remove-containers',
    async (url) => {
      const response = await fetch(url, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to remove containers. ' + response.statusText);
      const data: ApiResponse<CliOutput> = await response.json();
      if (data.status !== "success") {
        throw Error(data.message ?? "Failed to remove containers, no details");
      }
      return data;
    }
  );

  const [lastAsyncError, setLastAsyncError] = useState<string>();

  // Combine errors
  const error = ramError || diskError || dockerError;

  // Loading state — when data is undefined (initial load)
  const loading = !ramData && !ramError || !diskData && !diskError || !dockerData && !dockerError;

  const refreshAll = useCallback((): void => {
    mutateRam();
    mutateDisk();
    mutateDocker();
  }, []);

  const [startingDocker, setStartingDocker] = useState(false);

  const startDocker = useCallback(async (): Promise<void> => {
    setStartingDocker(true);
    try {
      await triggerStartDocker();
      mutateDocker(); // Re-validate Docker status after action
      setLastAsyncError(undefined);
    } catch (err: any) {
      setLastAsyncError(err.toString());
    }
    setStartingDocker(false);
  }, []);

  const [stoppingDocker, setStoppingDocker] = useState(false);

  const stopDocker = useCallback(async (): Promise<void> => {
    setStoppingDocker(true);
    try {
      await triggerStopDocker();
      mutateDocker(); // Re-validate Docker status after action
      setLastAsyncError(undefined);
    } catch (err: any) {
      setLastAsyncError(err.toString());
    }
    setStoppingDocker(false);
  }, []);

  const [creatingContainers, setCreatingContainers] = useState(false);

  const createContainers = async (): Promise<void> => {
    setCreatingContainers(true);
    try {
      await triggerCreateContainers();
      mutateDocker(); // Обновляем статус Docker после сборки
      setLastAsyncError(undefined);
    } catch (err: any) {
      setLastAsyncError(err.toString());
    }
    setCreatingContainers(false);
  };

  const [removingContainers, setRemovingContainers] = useState(false);

  const removeContainers = async (): Promise<void> => {
    setRemovingContainers(true);
    try {
      await triggerRemoveContainers();
      mutateDocker(); // Обновляем статус Docker после сборки
      setLastAsyncError(undefined);
    } catch (err: any) {
      setLastAsyncError(err.toString());
    }
    setRemovingContainers(false);
  };

  return {
    ramData,
    diskData,
    dockerData,
    loading,
    error,
    refreshAll,
    startDocker, startingDocker,
    stopDocker, stoppingDocker,
    createContainers, creatingContainers,
    removeContainers, removingContainers,
    lastAsyncError,
  };
};

const SystemInfo: React.FC = () => {
  const {
    ramData,
    diskData,
    dockerData,
    loading,
    error,
    refreshAll,
    startDocker,
    startingDocker,
    stopDocker,
    stoppingDocker,
    createContainers,
    creatingContainers,
    removeContainers,
    removingContainers,
    lastAsyncError,
  } = useSystemData();

  const ram = ramData?.status === 'success' ? ramData.data : null;
  const disk = diskData?.status === 'success' ? diskData.data : null;
  const docker = dockerData?.status === 'success' ? dockerData.data : null;

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <p>Loading system information...</p>
      </div>
    );
  }

  // Docker services table columns
  const columns = [
    {
      title: 'Service',
      dataIndex: 'Service',
      key: 'service',
      render: (service: string) => <span style={{ fontWeight: "bold" }}>{service}</span>
    },
    {
      title: 'Command',
      dataIndex: 'Command',
      key: 'command',
      ellipsis: true,
      render: (command: string) => <span style={{ fontSize: '12px' }}>{command}</span>
    },
    {
      title: 'Status',
      dataIndex: 'State',
      key: 'state',
      render: (state: string) => {
        let color: string;
        if (state === 'running' || state === 'up') color = 'green';
        else if (state === 'exited' || state === 'down' || state === "created") color = 'orange';
        else color = 'red';

        return <Tag color={color}>{state}</Tag>;
      }
    },
    {
      title: 'Ports',
      dataIndex: 'Ports',
      key: 'ports',
      ellipsis: true,
      render: (ports: string) => ports || 'N/A'
    },
    {
      title: 'Health',
      dataIndex: 'Health',
      key: 'health',
      render: (health: string) => {
        let color: string;
        if (health === 'healthy') color = 'green';
        else if (health === 'unhealthy') color = 'red';
        else if (health === 'starting') color = 'orange';
        else color = 'default';

        return <Tag color={color}>{health || 'N/A'}</Tag>;
      }
    },
  ];

  // Determine Docker overall status
  let dockerStatusText = 'Unknown';
  let dockerStatusColor = 'default';

  if (docker?.running === "yes") {
    dockerStatusText = `Running (${docker.running_count}/${docker.total_count} services)`;
    dockerStatusColor = 'green';
  } else if (docker?.running === "no") {
    dockerStatusText = 'Stopped';
    dockerStatusColor = 'red';
  }

  return (
    <Row gutter={[16, 16]}>
      {error && (
        <Col span={24}>
          <Alert
            message="Error"
            description={error.message || 'Failed to fetch system data'}
            type="error"
            showIcon
            action={
              <Button
                size="small"
                onClick={refreshAll}
                icon={<ReloadOutlined />}
              >
                Retry
              </Button>
            }
          />
        </Col>
      )}

      {lastAsyncError && (
        <Col span={24}>
          <Alert
            message="Error"
            description={lastAsyncError}
            type="error"
            showIcon
          />
        </Col>
      )}

      <Col xs={24} md={12}>
        <Card
          title="RAM Information"
          className="card"
        >
          {ram ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="Available RAM (free + cached files)"
                value={ram.available}
                precision={2}
                suffix="GB"
                valueStyle={{ color: '#3f8600' }}
              />
              <Statistic
                title="Total RAM"
                value={ram.total}
                precision={2}
                suffix="GB"
              />
              <Statistic
                title="RAM Available Percentage"
                value={ram.total != 0 ? ram.available / ram.total * 100 : "unknown"}
                precision={1}
                suffix="%"
                valueStyle={{
                  color: (ram.total != 0 && ram.available / ram.total > 0.2) ? '#3f8600' : '#cf1322'
                }}
              />
            </Space>
          ) : (
            <div>No data available</div>
          )}
        </Card>
      </Col>

      <Col xs={24} md={12}>
        <Card
          title="Disk Space Information"
          className="card"
        >
          {disk ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="Free Disk Space"
                value={disk.free}
                precision={2}
                suffix="GB"
                valueStyle={{ color: '#1890ff' }}
              />
              <Statistic
                title="Total Disk Space"
                value={disk.total}
                precision={2}
                suffix="GB"
              />
              <Statistic
                title="Disk Free Percentage"
                value={disk.percent_free}
                precision={1}
                suffix="%"
                valueStyle={{
                  color: disk.percent_free > 10 ? '#1890ff' : '#faad14'
                }}
              />
            </Space>
          ) : (
            <div>No data available</div>
          )}
        </Card>
      </Col>

      <div dangerouslySetInnerHTML={{__html: `<style>
        .docker-card .ant-card-head-title .ant-space { flex-wrap: wrap; white-space: wrap; }
        .docker-card .ant-card-head-wrapper { flex-wrap: wrap; }

        @media (max-width: 768px) {
          .docker-card .ant-card-extra .ant-space { flex-direction: column; }
        }
      </style>`}} />
      <Col span={24}>
        <Card
          className={"docker-card"}
          title={
            <Space className={"123"}>
              Docker Compose Status
              <Tag color={dockerStatusColor}>{dockerStatusText}</Tag>
              {docker?.services && (
                <Space>
                  <Tag color="green">{docker.services.filter(s => s.Health === 'healthy').length} Healthy</Tag>
                  <Tag color="red">{docker.services.filter(s => s.Health === 'unhealthy').length} Unhealthy</Tag>
                </Space>
              )}
            </Space>
          }
          extra={
            <Space wrap={true}>
              <Button
                type="primary"
                loading={startingDocker}
                icon={<PlayCircleOutlined />}
                onClick={startDocker}
                disabled={docker?.running !== "no" && (docker?.services.length ?? 0) > 0}
              >
                Start
              </Button>
              <Button
                danger
                loading={stoppingDocker}
                icon={<StopOutlined />}
                onClick={stopDocker}
                disabled={docker?.running !== "yes"}
              >
                Stop
              </Button>
              <Button
                type="default"
                loading={creatingContainers}
                icon={<PlusOutlined />}
                onClick={createContainers}
                disabled={docker?.running !== "no"}
              >
                Initialize
              </Button>
              <Button
                type="default"
                loading={removingContainers}
                icon={<MinusOutlined />}
                onClick={removeContainers}
                disabled={docker?.running !== "no"}
              >
                Remove
              </Button>
            </Space>
          }
        >
          {docker?.services && docker.services.length > 0 ? (
            <Table
              columns={columns}
              dataSource={docker.services}
              rowKey="Service"
              pagination={false}
              size="small"
            />
          ) : (
            <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>
              {dockerData?.status !== "success" && (dockerData?.message ?? 'No services found or Docker Compose file not available')}
              {dockerData?.status === "success" && 'No containers yet'}
            </div>
          )}
        </Card>
      </Col>
    </Row>
  );
};

export default SystemInfo;
