import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { api, apiBaseUrl } from '../../api';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  TextField,
  Button,
  CircularProgress,
  LinearProgress,
  Paper,
  Snackbar,
  Alert,
  Chip,
  OutlinedInput,
  Checkbox,
  ListItemText,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import LibraryAddIcon from '@mui/icons-material/LibraryAdd';

interface LoRAModel {
  name: string;
  model_name?: string;
  base_model?: string;
}

interface InferenceStatus {
  status: 'idle' | 'loading' | 'processing' | 'completed' | 'failed';
  progress: number;
  step: number;
  total_steps: number;
  message: string;
  image_id: string | null;
}

export interface InferencePanelProps {
  onImageGenerated?: (imageUrl: string) => void;
  onAddToCanvas?: (imageUrl: string) => Promise<void> | void;
}

const InferencePanel: React.FC<InferencePanelProps> = ({ onImageGenerated, onAddToCanvas }) => {
  const defaultBaseModel = 'runwayml/stable-diffusion-v1-5';
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [selectedBaseModel, setSelectedBaseModel] = useState(defaultBaseModel);
  const [selectedLoras, setSelectedLoras] = useState<string[]>([]);
  const [loraModels, setLoraModels] = useState<LoRAModel[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<InferenceStatus>({
    status: 'idle',
    progress: 0,
    step: 0,
    total_steps: 50,
    message: '',
    image_id: null,
  });
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error';
  } | null>(null);
  const [isAddingToCanvas, setIsAddingToCanvas] = useState(false);

  const pollingInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await api.get('/models');
        setLoraModels(response.data);
      } catch (err) {
        console.error('获取 LoRA 模型失败:', err);
      }
    };
    fetchModels();
  }, []);

  const baseModelOptions = useMemo(() => {
    const options = new Set<string>([defaultBaseModel]);
    loraModels
      .map((model) => model.base_model)
      .filter((modelName): modelName is string => Boolean(modelName))
      .forEach((modelName) => options.add(modelName));
    return Array.from(options);
  }, [defaultBaseModel, loraModels]);

  const visibleLoras = useMemo(
    () => loraModels.filter((model) => model.base_model === selectedBaseModel || !model.base_model),
    [loraModels, selectedBaseModel],
  );

  useEffect(() => {
    if (!baseModelOptions.includes(selectedBaseModel)) {
      setSelectedBaseModel(baseModelOptions[0]);
      return;
    }
    if (selectedBaseModel === defaultBaseModel && baseModelOptions.length > 1 && visibleLoras.length === 0) {
      setSelectedBaseModel(baseModelOptions[1]);
    }
  }, [baseModelOptions, defaultBaseModel, selectedBaseModel, visibleLoras.length]);

  useEffect(() => {
    const allowed = new Set(visibleLoras.map((model) => model.name));
    setSelectedLoras((prev) => {
      const next = prev.filter((name) => allowed.has(name));
      if (next.length === prev.length && next.every((name, index) => name === prev[index])) {
        return prev;
      }
      return next;
    });
  }, [visibleLoras]);

  const resolveLoraName = (name: string) =>
    loraModels.find((model) => model.name === name)?.model_name || name;

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await api.get('/generate/status');
        const newStatus: InferenceStatus = response.data;
        setStatus(newStatus);

        if (newStatus.status === 'completed') {
          if (pollingInterval.current) clearInterval(pollingInterval.current);
          setIsProcessing(false);
          const imageUrl = `${apiBaseUrl}/generate/image/${newStatus.image_id}?t=${new Date().getTime()}`;
          setGeneratedImage(imageUrl);
          setSnackbar({ open: true, message: '图片生成成功！', severity: 'success' });
          onImageGenerated?.(imageUrl);
        } else if (newStatus.status === 'failed') {
          if (pollingInterval.current) clearInterval(pollingInterval.current);
          setIsProcessing(false);
          setSnackbar({ open: true, message: newStatus.message, severity: 'error' });
        }
      } catch (error) {
        console.error('获取进度失败:', error);
        if (pollingInterval.current) clearInterval(pollingInterval.current);
        setIsProcessing(false);
      }
    };

    if (isProcessing) {
      pollingInterval.current = setInterval(pollStatus, 1500);
    } else if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
    }

    return () => {
      if (pollingInterval.current) clearInterval(pollingInterval.current);
    };
  }, [isProcessing, onImageGenerated]);

  const handleGenerateImage = async () => {
    if (!prompt) {
      setSnackbar({ open: true, message: '请输入提示词。', severity: 'error' });
      return;
    }

    setIsProcessing(true);
    setGeneratedImage(null);
    setStatus((prev) => ({ ...prev, status: 'loading', message: '正在向服务器发送请求...' }));

    try {
      await api.post('/generate', {
        prompt,
        negative_prompt: negativePrompt,
        base_model: selectedBaseModel,
        lora_models: selectedLoras,
      });
    } catch (error) {
      let message = '发生未知错误。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data.detail || error.response.data.message || message;
      }
      setSnackbar({ open: true, message, severity: 'error' });
      setIsProcessing(false);
    }
  };

  const handleAddToCanvas = async () => {
    if (!generatedImage || !onAddToCanvas || isAddingToCanvas) {
      return;
    }

    try {
      setIsAddingToCanvas(true);
      await onAddToCanvas(generatedImage);
      setSnackbar({ open: true, message: '已添加到画布！', severity: 'success' });
    } catch (error) {
      const message = error instanceof Error ? error.message : '添加到画布失败。';
      setSnackbar({ open: true, message, severity: 'error' });
    } finally {
      setIsAddingToCanvas(false);
    }
  };

  const handleDownload = () => {
    if (generatedImage) {
      const link = document.createElement('a');
      link.href = generatedImage;
      link.download = `生成图片_${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const isTaskActive = status.status === 'loading' || status.status === 'processing';

  return (
    <>
      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Card sx={{ minWidth: { xs: '100%', md: 360 } }}>
            <CardContent>
              <Typography variant="h5" component="h2" gutterBottom>
                参数设置
              </Typography>
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel id="base-model-select-label">基础模型</InputLabel>
                <Select
                  labelId="base-model-select-label"
                  value={selectedBaseModel}
                  label="基础模型"
                  onChange={(e) => setSelectedBaseModel(e.target.value)}
                  disabled={isProcessing}
                >
                  {baseModelOptions.map((modelName) => (
                    <MenuItem key={modelName} value={modelName}>
                      {modelName}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel id="lora-select-label">LoRA 模型（可选）</InputLabel>
                <Select
                  labelId="lora-select-label"
                  multiple
                  value={selectedLoras}
                  input={<OutlinedInput label="LoRA 模型（可选）" />}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={resolveLoraName(value)} />
                      ))}
                    </Box>
                  )}
                  onChange={(event) => {
                    const value = event.target.value;
                    setSelectedLoras(typeof value === 'string' ? value.split(',') : value);
                  }}
                  disabled={isProcessing}
                >
                  {visibleLoras.length === 0 && (
                    <MenuItem disabled>该基础模型暂无可用 LoRA</MenuItem>
                  )}
                  {visibleLoras.map((model) => (
                    <MenuItem key={model.name} value={model.name}>
                      <Checkbox checked={selectedLoras.includes(model.name)} />
                      <ListItemText primary={model.model_name || model.name} />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Box sx={{ mt: 3 }}>
                <TextField
                  fullWidth
                  label="正向提示词（例如：'a beautiful landscape painting'）"
                  variant="outlined"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  disabled={isProcessing}
                  multiline
                  rows={4}
                />
              </Box>

              <Box sx={{ mt: 2 }}>
                <TextField
                  fullWidth
                  label="反向提示词（例如：'blurry, low quality'）"
                  variant="outlined"
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  disabled={isProcessing}
                  multiline
                  rows={3}
                />
              </Box>

              <Box sx={{ mt: 3, textAlign: 'center' }}>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  onClick={handleGenerateImage}
                  disabled={isProcessing}
                  sx={{ width: '100%' }}
                >
                  {isProcessing ? <CircularProgress size={24} color="inherit" /> : '生成图片'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%', minWidth: { xs: '100%', md: 420 } }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <Typography variant="h5" component="h2" gutterBottom>
                生成结果
              </Typography>
              <Box
                sx={{
                  flexGrow: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: '#F8F9FA',
                  borderRadius: 1,
                  minHeight: 320,
                }}
              >
                {isTaskActive && (
                  <Box sx={{ textAlign: 'center' }}>
                    <CircularProgress size={60} />
                    <Typography variant="h6" sx={{ mt: 2 }}>
                      {status.message}
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={status.progress}
                      sx={{ width: '80%', margin: '16px auto' }}
                    />
                  </Box>
                )}
                {!isTaskActive && generatedImage && (
                  <Paper elevation={3} sx={{ display: 'inline-block', lineHeight: 0 }}>
                    <img
                      src={generatedImage}
                      alt="Stable Diffusion 生成"
                      style={{ maxWidth: '100%', maxHeight: '60vh', borderRadius: '4px' }}
                    />
                  </Paper>
                )}
                {!isTaskActive && !generatedImage && (
                  <Typography variant="body1" color="text.secondary">
                    生成的图片会显示在这里
                  </Typography>
                )}
              </Box>
              {generatedImage && !isTaskActive && (
                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
                  <Button variant="outlined" startIcon={<DownloadIcon />} onClick={handleDownload}>
                    下载
                  </Button>
                  {onAddToCanvas && (
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<LibraryAddIcon />}
                      onClick={handleAddToCanvas}
                      disabled={isAddingToCanvas}
                    >
                      {isAddingToCanvas ? '添加中...' : '添加到画布'}
                    </Button>
                  )}
                  <Button
                    variant="outlined"
                    startIcon={<ContentCopyIcon />}
                    onClick={() => navigator.clipboard.writeText('暂无种子')}
                  >
                    复制种子
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {snackbar && (
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={() => setSnackbar(null)} severity={snackbar.severity} sx={{ width: '100%' }}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      )}
    </>
  );
};

export default InferencePanel;
