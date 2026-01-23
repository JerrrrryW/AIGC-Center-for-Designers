
import React, { useEffect, useMemo, useState } from 'react';
import {
  AppBar,
  Box,
  Button,
  CssBaseline,
  Divider,
  Drawer,
  Collapse,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Stack,
  TextField,
  Toolbar,
  Typography,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { Link, useLocation } from 'react-router-dom';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import ImagesearchRollerIcon from '@mui/icons-material/ImagesearchRoller';
import StorageIcon from '@mui/icons-material/Storage';
import BrushIcon from '@mui/icons-material/Brush';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { api } from '../api';

const drawerWidth = 240;
const LLM_CONFIG_STORAGE_KEY = 'llm-config';

type LlmConfigState = {
  apiKey: string;
  baseUrl: string;
  model: string;
  temperature: string;
  imageApiKey: string;
  imageBaseUrl: string;
  imageModel: string;
};

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [llmOpen, setLlmOpen] = useState(false);
  const [llmConfig, setLlmConfig] = useState<LlmConfigState>({
    apiKey: '',
    baseUrl: '',
    model: '',
    temperature: '',
    imageApiKey: '',
    imageBaseUrl: '',
    imageModel: '',
  });
  const [llmStatus, setLlmStatus] = useState<string | null>(null);
  const [llmSaving, setLlmSaving] = useState(false);
  const location = useLocation();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'AI 场景生成', path: '/scene', icon: <AutoAwesomeIcon /> },
    { text: 'LoRA 训练', path: '/', icon: <ModelTrainingIcon /> },
    { text: '推理生成', path: '/inference', icon: <ImagesearchRollerIcon /> },
    { text: '训练模型', path: '/models', icon: <StorageIcon /> },
    { text: '画布', path: '/canvas', icon: <BrushIcon /> },
  ];
  const isCanvasRoute = location.pathname === '/canvas';

  useEffect(() => {
    const stored = localStorage.getItem(LLM_CONFIG_STORAGE_KEY);
    if (!stored) return;
    try {
      const parsed = JSON.parse(stored) as Partial<LlmConfigState>;
      setLlmConfig((prev) => ({ ...prev, ...parsed }));
    } catch {
      // ignore invalid local cache
    }
  }, []);

  const normalizedPayload = useMemo(() => {
    const temperatureRaw = llmConfig.temperature.trim();
    const temperature = temperatureRaw ? Number(temperatureRaw) : null;
    return {
      api_key: llmConfig.apiKey.trim() || null,
      base_url: llmConfig.baseUrl.trim() || null,
      model: llmConfig.model.trim() || null,
      temperature: Number.isFinite(temperature) ? temperature : null,
      image_api_key: llmConfig.imageApiKey.trim() || null,
      image_base_url: llmConfig.imageBaseUrl.trim() || null,
      image_model: llmConfig.imageModel.trim() || null,
    };
  }, [llmConfig]);

  const handleSaveLlmConfig = async () => {
    setLlmStatus(null);
    const tempValue = llmConfig.temperature.trim();
    if (tempValue && Number.isNaN(Number(tempValue))) {
      setLlmStatus('温度需要填写数字');
      return;
    }
    setLlmSaving(true);
    try {
      await api.post('/llm-config', normalizedPayload);
      localStorage.setItem(LLM_CONFIG_STORAGE_KEY, JSON.stringify(llmConfig));
      setLlmStatus('配置已保存');
    } catch (error) {
      setLlmStatus('保存失败，请检查后端服务');
    } finally {
      setLlmSaving(false);
    }
  };

  const drawer = (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Toolbar />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: 'rgba(0, 123, 255, 0.08)',
                  borderLeft: '3px solid #007BFF',
                  '&:hover': {
                    backgroundColor: 'rgba(0, 123, 255, 0.12)',
                  },
                },
              }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Box sx={{ mt: 'auto', p: 2 }}>
        <Divider sx={{ mb: 1.5 }} />
        <ListItemButton
          onClick={() => setLlmOpen((prev) => !prev)}
          sx={{ px: 1, borderRadius: 1 }}
        >
          <ListItemText primary="LLM 配置" primaryTypographyProps={{ variant: 'body2' }} />
          {llmOpen ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </ListItemButton>
        <Collapse in={llmOpen} timeout="auto" unmountOnExit>
          <Stack spacing={1.5} sx={{ mt: 1 }}>
            <TextField
              label="API Key"
              type="password"
              size="small"
              value={llmConfig.apiKey}
              onChange={(e) => setLlmConfig((prev) => ({ ...prev, apiKey: e.target.value }))}
              fullWidth
            />
            <TextField
              label="Base URL"
              size="small"
              placeholder="https://api.siliconflow.cn/v1/chat/completions"
              value={llmConfig.baseUrl}
              onChange={(e) => setLlmConfig((prev) => ({ ...prev, baseUrl: e.target.value }))}
              fullWidth
            />
            <TextField
              label="Model"
              size="small"
              placeholder="Qwen/Qwen2.5-7B-Instruct"
              value={llmConfig.model}
              onChange={(e) => setLlmConfig((prev) => ({ ...prev, model: e.target.value }))}
              fullWidth
            />
            <TextField
              label="Temperature"
              size="small"
              placeholder="0.4"
              value={llmConfig.temperature}
              onChange={(e) => setLlmConfig((prev) => ({ ...prev, temperature: e.target.value }))}
              fullWidth
            />
            <Divider />
            <Typography variant="caption" color="text.secondary">
              生图配置（可选，默认复用 LLM Key）
            </Typography>
            <TextField
              label="Image API Key"
              type="password"
              size="small"
              value={llmConfig.imageApiKey}
              onChange={(e) => setLlmConfig((prev) => ({ ...prev, imageApiKey: e.target.value }))}
              fullWidth
            />
            <TextField
              label="Image Base URL"
              size="small"
              placeholder="https://api.siliconflow.cn/v1"
              value={llmConfig.imageBaseUrl}
              onChange={(e) => setLlmConfig((prev) => ({ ...prev, imageBaseUrl: e.target.value }))}
              fullWidth
            />
            <TextField
              label="Image Model"
              size="small"
              placeholder="Qwen/Qwen-Image"
              value={llmConfig.imageModel}
              onChange={(e) => setLlmConfig((prev) => ({ ...prev, imageModel: e.target.value }))}
              fullWidth
            />
            <Button
              variant="contained"
              size="small"
              onClick={handleSaveLlmConfig}
              disabled={llmSaving}
            >
              {llmSaving ? '保存中...' : '保存配置'}
            </Button>
            {llmStatus ? (
              <Typography variant="caption" color={llmStatus.includes('失败') ? 'error' : 'text.secondary'}>
                {llmStatus}
              </Typography>
            ) : null}
          </Stack>
        </Collapse>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="打开侧边栏"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            DHUX AIGC 中心
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        component="nav"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
        variant="permanent"
        anchor="left"
      >
        {drawer}
      </Drawer>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          bgcolor: 'background.default',
          p: 3,
          width: '100%',
        }}
      >
        <Toolbar />
        <Box
          sx={{
            maxWidth: isCanvasRoute ? '100%' : '1920px',
            margin: isCanvasRoute ? 0 : '0 auto',
            height: isCanvasRoute ? '100%' : 'auto',
          }}
        >
          {children}
        </Box>
      </Box>
    </Box>
  );
};

export default Layout;
