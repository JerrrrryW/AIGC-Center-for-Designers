
import React, { useState } from 'react';
import {
  AppBar,
  Box,
  CssBaseline,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
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

const drawerWidth = 240;

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'AI 场景生成', path: '/scene/chat', icon: <AutoAwesomeIcon /> },
    { text: 'LoRA 训练', path: '/', icon: <ModelTrainingIcon /> },
    { text: '推理生成', path: '/inference', icon: <ImagesearchRollerIcon /> },
    { text: '训练模型', path: '/models', icon: <StorageIcon /> },
    { text: '画布', path: '/canvas', icon: <BrushIcon /> },
  ];
  const isCanvasRoute = location.pathname === '/canvas';

  const drawer = (
    <div>
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
    </div>
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
            maxWidth: isCanvasRoute ? '100%' : '1280px',
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
