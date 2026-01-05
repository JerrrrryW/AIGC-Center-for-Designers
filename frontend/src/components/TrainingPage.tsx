import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { api } from '../api';
import { 
  Container, Typography, Card, CardContent, Button, TextField, Slider, Box, Grid, 
  CircularProgress, Snackbar, Alert, LinearProgress, Tooltip, IconButton, Select, MenuItem, FormControl, InputLabel,
  FormControlLabel, Checkbox
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import DeleteIcon from '@mui/icons-material/Delete';

interface TrainingStatus {
  status: 'idle' | 'initializing' | 'loading_models' | 'training' | 'completed' | 'failed';
  progress: number;
  message: string;
}

interface CaptionStatus {
  status: 'idle' | 'loading' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  results?: Record<string, string>;
}

const TrainingPage: React.FC = () => {
  // Form state
  const [files, setFiles] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [baseModel, setBaseModel] = useState('runwayml/stable-diffusion-v1-5');
  const [modelName, setModelName] = useState('');
  const [instancePrompt, setInstancePrompt] = useState('');
  const [steps, setSteps] = useState<number>(500);
  const [learningRate, setLearningRate] = useState<number>(1e-4);
  const [resolution, setResolution] = useState<number>(512);
  const [trainBatchSize, setTrainBatchSize] = useState<number>(1);
  const [captions, setCaptions] = useState<Record<string, string>>({});
  const [useCaptions, setUseCaptions] = useState(false);

  // UI and Status state
  const [isProcessing, setIsProcessing] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({ status: 'idle', progress: 0, message: '' });
  const [isCaptioning, setIsCaptioning] = useState(false);
  const [captionStatus, setCaptionStatus] = useState<CaptionStatus>({
    status: 'idle',
    progress: 0,
    message: '',
    results: {},
  });
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' } | null>(null);
  const pollingInterval = useRef<NodeJS.Timeout | null>(null);
  const captionPollingInterval = useRef<NodeJS.Timeout | null>(null);

  const handleFileChange = (newFiles: File[]) => {
    setFiles(prevFiles => [...prevFiles, ...newFiles]);
    const newPreviews = newFiles.map(file => URL.createObjectURL(file));
    setImagePreviews(prevPreviews => [...prevPreviews, ...newPreviews]);
    if (newFiles.length > 0) {
      setUseCaptions(false);
    }
  };

  const onDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const droppedFiles = Array.from(event.dataTransfer.files);
    handleFileChange(droppedFiles);
  }, []);

  const onDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleRemoveImage = (index: number) => {
    const fileName = files[index]?.name;
    setFiles(files.filter((_, i) => i !== index));
    setImagePreviews(imagePreviews.filter((_, i) => i !== index));
    if (fileName) {
      setCaptions((prev) => {
        const next = { ...prev };
        delete next[fileName];
        return next;
      });
    }
  };

  // Polling effect
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await api.get('/train/status');
        const newStatus: TrainingStatus = response.data;
        setTrainingStatus(newStatus);

        if (newStatus.status === 'completed' || newStatus.status === 'failed') {
          if (pollingInterval.current) clearInterval(pollingInterval.current);
          setIsProcessing(false);
          const severity = newStatus.status === 'completed' ? 'success' : 'error';
          setSnackbar({ open: true, message: newStatus.message, severity: severity });
        }
      } catch (error) {
        console.error("轮询训练状态失败:", error);
        if (pollingInterval.current) clearInterval(pollingInterval.current);
        setIsProcessing(false);
      }
    };

    if (isProcessing && (trainingStatus.status === 'initializing' || trainingStatus.status === 'loading_models' || trainingStatus.status === 'training')) {
        pollingInterval.current = setInterval(pollStatus, 2000);
    } else {
        if (pollingInterval.current) clearInterval(pollingInterval.current);
    }

    return () => {
      if (pollingInterval.current) clearInterval(pollingInterval.current);
    };
  }, [isProcessing, trainingStatus.status]);

  useEffect(() => {
    const pollCaptionStatus = async () => {
      try {
        const response = await api.get('/caption/status');
        const newStatus: CaptionStatus = response.data;
        setCaptionStatus(newStatus);

        if (newStatus.status === 'completed') {
          if (captionPollingInterval.current) clearInterval(captionPollingInterval.current);
          setIsCaptioning(false);
          if (newStatus.results) {
            setCaptions(() => {
              const allowed = new Set(files.map((file) => file.name));
              return Object.fromEntries(
                Object.entries(newStatus.results).filter(([name]) => allowed.has(name)),
              );
            });
          }
          setSnackbar({ open: true, message: newStatus.message || '描述生成完成。', severity: 'success' });
        } else if (newStatus.status === 'failed') {
          if (captionPollingInterval.current) clearInterval(captionPollingInterval.current);
          setIsCaptioning(false);
          setSnackbar({ open: true, message: newStatus.message || '描述生成失败。', severity: 'error' });
        }
      } catch (error) {
        console.error("轮询描述状态失败:", error);
        if (captionPollingInterval.current) clearInterval(captionPollingInterval.current);
        setIsCaptioning(false);
      }
    };

    if (isCaptioning) {
      captionPollingInterval.current = setInterval(pollCaptionStatus, 1500);
    } else {
      if (captionPollingInterval.current) clearInterval(captionPollingInterval.current);
    }

    return () => {
      if (captionPollingInterval.current) clearInterval(captionPollingInterval.current);
    };
  }, [files, isCaptioning]);

  useEffect(() => {
    if (useCaptions && Object.keys(captions).length === 0) {
      setUseCaptions(false);
    }
  }, [captions, useCaptions]);

  const handleStartTraining = async () => {
    if (files.length === 0) {
      setSnackbar({ open: true, message: '请先选择图片。', severity: 'error' });
      return;
    }
    if (useCaptions && Object.keys(captions).length === 0) {
      setSnackbar({ open: true, message: '请先生成或输入描述。', severity: 'error' });
      return;
    }

    setIsProcessing(true);
    setTrainingStatus({ status: 'initializing', progress: 0, message: '正在发送请求...' });

    const formData = new FormData();
    files.forEach(file => formData.append('images', file));
    formData.append('modelName', modelName);
    formData.append('baseModel', baseModel);
    formData.append('instancePrompt', instancePrompt);
    formData.append('steps', steps.toString());
    formData.append('learningRate', learningRate.toString());
    formData.append('resolution', resolution.toString());
    formData.append('trainBatchSize', trainBatchSize.toString());
    formData.append('useCaptions', useCaptions ? 'true' : 'false');
    if (useCaptions) {
      const captionsPayload: Record<string, string> = {};
      files.forEach((file) => {
        captionsPayload[file.name] = captions[file.name] || '';
      });
      formData.append('captions', JSON.stringify(captionsPayload));
    }

    try {
      const response = await api.post('/train', formData);
      setSnackbar({ open: true, message: response.data.message, severity: 'success' });
    } catch (error) {
      let message = '发生未知错误。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data.detail || error.response.data.message || message;
      }
      setSnackbar({ open: true, message, severity: 'error' });
      setIsProcessing(false);
    }
  };

  const handleCancelTraining = async () => {
    try {
      const response = await api.post('/train/terminate');
      setSnackbar({ open: true, message: response.data.message, severity: 'success' });
    } catch (error) {
      let message = '发送终止训练请求失败。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data.detail || error.response.data.message || message;
      }
      setSnackbar({ open: true, message, severity: 'error' });
    }
  };

  const handleStartCaptioning = async () => {
    if (files.length === 0) {
      setSnackbar({ open: true, message: '请先选择图片。', severity: 'error' });
      return;
    }

    setIsCaptioning(true);
    setCaptionStatus({ status: 'loading', progress: 0, message: '正在发送描述请求...', results: {} });

    const formData = new FormData();
    files.forEach(file => formData.append('images', file));

    try {
      const response = await api.post('/caption', formData);
      setSnackbar({ open: true, message: response.data.message, severity: 'success' });
    } catch (error) {
      let message = '发生未知错误。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data.detail || error.response.data.message || message;
      }
      setSnackbar({ open: true, message, severity: 'error' });
      setIsCaptioning(false);
    }
  };

  const isTrainingActive = trainingStatus.status !== 'idle' && trainingStatus.status !== 'completed' && trainingStatus.status !== 'failed';

  return (
    <Container maxWidth={false} sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        LoRA 模型训练
      </Typography>

      {isTrainingActive && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>{trainingStatus.message}</Typography>
            <LinearProgress variant="determinate" value={trainingStatus.progress} />
          </CardContent>
        </Card>
      )}

      <Grid container spacing={4}>
        <Grid item xs={12} md={5}>
          <Card sx={{ maxWidth: '500px' }}>
            <CardContent>
              <Typography variant="h2" gutterBottom>1. 上传图片</Typography>
              <Box
                sx={{
                  border: '2px dashed grey',
                  borderRadius: 2,
                  p: 3,
                  textAlign: 'center',
                  cursor: 'pointer',
                  backgroundColor: '#F8F9FA'
                }}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onClick={() => document.getElementById('file-input')?.click()}
              >
                <input 
                  id="file-input" 
                  type="file" 
                  multiple 
                  accept="image/*" 
                  style={{ display: 'none' }} 
                  onChange={(e) => handleFileChange(Array.from(e.target.files || []))}
                />
                <UploadFileIcon sx={{ fontSize: 48, color: 'grey.500' }} />
                <Typography>拖拽图片到这里，或点击选择文件</Typography>
              </Box>
              <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                <Button
                  variant="outlined"
                  onClick={handleStartCaptioning}
                  disabled={isCaptioning || files.length === 0 || isProcessing}
                >
                  {isCaptioning ? '生成描述中...' : '自动生成描述'}
                </Button>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={useCaptions}
                      onChange={(event) => setUseCaptions(event.target.checked)}
                      disabled={Object.keys(captions).length === 0 || isProcessing}
                    />
                  }
                  label="使用每张图片的描述进行训练"
                />
              </Box>
              {(isCaptioning || captionStatus.status === 'processing') && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    {captionStatus.message || '正在生成描述...'}
                  </Typography>
                  <LinearProgress variant="determinate" value={captionStatus.progress} />
                </Box>
              )}
              {imagePreviews.length > 0 && (
                <Box sx={{ 
                  mt: 2, 
                  display: 'grid', 
                  gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', 
                  gap: '16px',
                  maxHeight: '360px', // Approx 3 rows (100px image + 16px gap) * 3
                  overflowY: 'auto',
                  pr: 1 // Padding to avoid scrollbar overlapping content
                }}>
                  {imagePreviews.map((src, index) => {
                    const fileName = files[index]?.name;
                    return (
                      <Box key={src} sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        <Box sx={{ position: 'relative' }}>
                          <img src={src} alt="预览" width="100%" height="100%" style={{ objectFit: 'cover', borderRadius: '8px' }} />
                          <IconButton size="small" onClick={() => handleRemoveImage(index)} sx={{ position: 'absolute', top: 0, right: 0, backgroundColor: 'rgba(255,255,255,0.7)' }}>
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Box>
                        <TextField
                          size="small"
                          placeholder="图片描述（可选）"
                          value={fileName ? (captions[fileName] || '') : ''}
                          onChange={(event) => {
                            if (!fileName) {
                              return;
                            }
                            const nextValue = event.target.value;
                            setCaptions((prev) => ({ ...prev, [fileName]: nextValue }));
                          }}
                          disabled={isProcessing}
                        />
                      </Box>
                    );
                  })}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Typography variant="h2" gutterBottom>2. 设置参数</Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>基础模型 ID</InputLabel>
                <Select value={baseModel} label="基础模型 ID" onChange={(e) => setBaseModel(e.target.value)} disabled={isProcessing}>
                  <MenuItem value="runwayml/stable-diffusion-v1-5">runwayml/stable-diffusion-v1-5</MenuItem>
                  {/* Add other models here if available */}
                </Select>
              </FormControl>
              <TextField fullWidth label="模型名称（例如：'MyDog'）" variant="outlined" sx={{ mb: 2 }} value={modelName} onChange={(e) => setModelName(e.target.value)} disabled={isProcessing} />
              <TextField fullWidth label="训练提示词（例如：'a photo of sks dog'）" variant="outlined" sx={{ mb: 2 }} value={instancePrompt} onChange={(e) => setInstancePrompt(e.target.value)} disabled={isProcessing} />
              
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Tooltip title="设置训练步数。步数越多效果可能更好，但训练耗时更长。">
                  <IconButton size="small"><HelpOutlineIcon fontSize="small" /></IconButton>
                </Tooltip>
                <Typography sx={{ flexShrink: 0, mr: 2 }}>训练步数</Typography>
                <Slider value={steps} onChange={(_, newValue) => setSteps(newValue as number)} aria-label="训练步数" step={100} marks min={100} max={2000} disabled={isProcessing} />
                <Typography sx={{ ml: 2, width: '70px' }}>{steps}</Typography>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Tooltip title="学习率控制每次更新权重的幅度。学习率更小会更精细，但需要更多训练步数。">
                  <IconButton size="small"><HelpOutlineIcon fontSize="small" /></IconButton>
                </Tooltip>
                <Typography sx={{ flexShrink: 0, mr: 2 }}>学习率</Typography>
                <Slider value={learningRate} onChange={(_, newValue) => setLearningRate(newValue as number)} aria-label="学习率" step={1e-5} min={1e-5} max={1e-3} scale={(x) => x} disabled={isProcessing} />
                <Typography sx={{ ml: 2, width: '70px' }}>{learningRate.toExponential(1)}</Typography>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Tooltip title="设置训练图像分辨率。分辨率越高细节越多，但训练时间与内存占用显著增加。建议与基础模型常用尺寸一致（如 v1.5 为 512px）。">
                  <IconButton size="small"><HelpOutlineIcon fontSize="small" /></IconButton>
                </Tooltip>
                <Typography sx={{ flexShrink: 0, mr: 2 }}>分辨率</Typography>
                <Slider value={resolution} onChange={(_, newValue) => setResolution(newValue as number)} aria-label="分辨率" step={128} marks min={512} max={1024} disabled={isProcessing} />
                <Typography sx={{ ml: 2, width: '70px' }}>{resolution}px</Typography>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Tooltip title="每次训练看到的图片数量。内存紧张的 Mac 建议保持为 1。增大会加快训练但显著增加内存占用，可能导致失败。">
                  <IconButton size="small"><HelpOutlineIcon fontSize="small" /></IconButton>
                </Tooltip>
                <Typography sx={{ flexShrink: 0, mr: 2 }}>批大小</Typography>
                <Slider value={trainBatchSize} onChange={(_, newValue) => setTrainBatchSize(newValue as number)} aria-label="批大小" step={1} marks min={1} max={8} disabled={isProcessing} />
                <Typography sx={{ ml: 2, width: '70px' }}>{trainBatchSize}</Typography>
              </Box>

            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 4, textAlign: 'center' }}>
        {!isTrainingActive ? (
          <Button variant="contained" color="primary" size="large" onClick={handleStartTraining} disabled={isProcessing} sx={{ minWidth: 150 }}>
            {isProcessing ? <><CircularProgress size={24} color="inherit" sx={{ mr: 1 }} /> 训练中...</> : '开始训练'}
          </Button>
        ) : (
          <Button variant="contained" color="error" size="large" onClick={handleCancelTraining} disabled={!isTrainingActive}>
            取消训练
          </Button>
        )}
      </Box>

      {snackbar && (
        <Snackbar open={snackbar.open} autoHideDuration={6000} onClose={() => setSnackbar(null)} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
          <Alert onClose={() => setSnackbar(null)} severity={snackbar.severity} sx={{ width: '100%' }}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      )}
    </Container>
  );
};

export default TrainingPage;
