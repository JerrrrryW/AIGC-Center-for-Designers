import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { api, apiBaseUrl } from '../api';
import {
  Container, Typography, Card, CardContent, Grid, Box, CircularProgress, Alert, Button, Dialog, 
  DialogActions, DialogContent, DialogContentText, DialogTitle, CardActions, CardMedia, TextField, IconButton
} from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';

interface LoRAModel {
  name: string; // This is the folder name, used as ID
  model_name?: string; // This is the user-defined name
  base_model?: string;
  prompt?: string;
  creation_time?: string;
  thumbnail_url?: string; // Optional thumbnail
}

const PLACEHOLDER_IMAGE = `data:image/svg+xml;utf8,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="140" viewBox="0 0 300 140"><rect width="300" height="140" fill="#e0e0e0"/><text x="50%" y="50%" text-anchor="middle" fill="#757575" font-size="20" font-family="Arial" dy=".35em">暂无预览</text></svg>',
)}`;

const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<LoRAModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [selectedModel, setSelectedModel] = useState<LoRAModel | null>(null);
  const [editingModel, setEditingModel] = useState<string | null>(null);
  const [newName, setNewName] = useState('');

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      const response = await api.get('/models');
      setModels(response.data);
    } catch (err) {
      setError('获取模型失败。请确认后端服务已启动。');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleDownload = (modelName: string) => {
    window.open(`${apiBaseUrl}/models/download/${modelName}`);
  };

  const handleDeleteClick = (model: LoRAModel) => {
    setSelectedModel(model);
    setOpenDeleteDialog(true);
  };

  const handleDeleteConfirm = async () => {
    if (selectedModel) {
      try {
        await api.delete(`/models/delete/${selectedModel.name}`);
        setOpenDeleteDialog(false);
        setSelectedModel(null);
        fetchModels(); // Refresh the model list
      } catch (err) {
        setError(`删除模型 ${selectedModel.name} 失败。`);
        console.error(err);
      }
    }
  };

  const handleRenameStart = (model: LoRAModel) => {
    setEditingModel(model.name);
    setNewName(model.model_name || model.name);
  };

  const handleRenameCancel = () => {
    setEditingModel(null);
    setNewName('');
  };

  const handleRenameSave = async (originalName: string) => {
    try {
      await api.put(`/models/rename/${originalName}`, { new_name: newName });
      setEditingModel(null);
      setNewName('');
      fetchModels();
    } catch (err) {
      setError('重命名模型失败。');
      console.error(err);
    }
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        已训练的 LoRA 模型
      </Typography>

      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mt: 4 }}>
          {error}
        </Alert>
      )}

      {!isLoading && !error && models.length === 0 && (
        <Typography sx={{ mt: 4 }}>暂无已训练模型。</Typography>
      )}

      <Grid container spacing={3} sx={{ mt: 2 }}>
        {models.map((model) => (
          <Grid item xs={12} sm={6} md={4} key={model.name}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardMedia
                component="img"
                height="140"
                image={model.thumbnail_url || PLACEHOLDER_IMAGE}
                alt={`预览：${model.model_name || model.name}`}
              />
              <CardContent sx={{ flexGrow: 1 }}>
                {editingModel === model.name ? (
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <TextField 
                      value={newName} 
                      onChange={(e) => setNewName(e.target.value)} 
                      size="small" 
                      variant="outlined"
                      fullWidth
                    />
                    <IconButton onClick={() => handleRenameSave(model.name)} size="small"><SaveIcon /></IconButton>
                    <IconButton onClick={handleRenameCancel} size="small"><CancelIcon /></IconButton>
                  </Box>
                ) : (
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Typography variant="h6" component="div">
                      {model.model_name || model.name}
                    </Typography>
                    <IconButton onClick={() => handleRenameStart(model)} size="small"><EditIcon /></IconButton>
                  </Box>
                )}
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  ID：{model.name}
                </Typography>
                <Typography sx={{ mt: 1.5 }} color="text.secondary">
                  基础模型：<strong>{model.base_model || '未知'}</strong>
                </Typography>
                <Typography sx={{ mt: 1.5 }} color="text.secondary">
                  实例提示词：<strong>{model.prompt || '暂无'}</strong>
                </Typography>
                <Typography sx={{ mt: 1 }} color="text.secondary">
                  创建时间：{model.creation_time ? new Date(model.creation_time).toLocaleString() : '未知'}
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => handleDownload(model.name)}>下载</Button>
                <Button size="small" color="error" onClick={() => handleDeleteClick(model)}>删除</Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={openDeleteDialog}
        onClose={() => setOpenDeleteDialog(false)}
      >
        <DialogTitle>确认删除</DialogTitle>
        <DialogContent>
          <DialogContentText>
            确定要删除模型“{selectedModel?.model_name}”吗？此操作无法撤销。
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDeleteDialog(false)}>取消</Button>
          <Button onClick={handleDeleteConfirm} color="error" autoFocus>
            删除
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ModelsPage;
