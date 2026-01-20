import React, { useCallback, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Grid,
  LinearProgress,
  Snackbar,
  Alert,
  TextField,
  Typography,
  Chip,
  Stack,
  ToggleButtonGroup,
  ToggleButton,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Paper,
} from '@mui/material';
import axios from 'axios';
import { api } from '../api';

type Mode = 'single' | 'layered' | 'compose';

type LayerItem = {
  id: string;
  name: string;
  layer_type: string;
  image_base64: string;
  width?: number;
  height?: number;
  placement?: { x?: number; y?: number; z_index?: number };
};

const SceneGenerationPage: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [mode, setMode] = useState<Mode>('single');
  const [aspectRatio, setAspectRatio] = useState('3:4');
  const [status, setStatus] = useState<string>('');
  const [layoutTips, setLayoutTips] = useState<string[]>([]);
  const [formItems, setFormItems] = useState<Array<{ title: string; options: string[] }>>([]);
  const [selectedOptions, setSelectedOptions] = useState<Record<string, string[]>>({});
  const [positivePrompt, setPositivePrompt] = useState('');
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [layers, setLayers] = useState<LayerItem[]>([]);
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'bot'; text: string }>>([
    { role: 'bot', text: '你好！说说你想要生成的场景，我会给出表单和示例帮助你细化。' },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error';
  } | null>(null);

  const handleGenerate = async (incomingPrompt?: string) => {
    const effectivePrompt = (incomingPrompt ?? prompt).trim();
    if (!effectivePrompt) {
      setSnackbar({ open: true, message: '请输入你的场景想法。', severity: 'error' });
      return;
    }
    setIsLoading(true);
    setStatus('正在生成表单与提示词...');
    setLayoutTips([]);
    setFormItems([]);
    setSelectedOptions({});
    setImageUrl(null);
    setLayers([]);
    try {
      const layoutRes = await api.post('/generate-layout', { prompt: effectivePrompt });
      const form = layoutRes.data?.form || [];
      setFormItems(form);
      const tips = form.flatMap((item: any) => item?.options || []);
      setLayoutTips(tips);

      setStatus('正在生成正向提示词...');
      const textRes = await api.post('/generate-text', { prompt: effectivePrompt });
      const positive = textRes.data?.positive || effectivePrompt;
      setPositivePrompt(positive);

      if (mode === 'single') {
        setStatus('正在生成图片...');
        const imgRes = await api.post('/generate-image', {
          prompt: positive,
          width: 768,
          height: 768,
        });
        const base64 = imgRes.data?.image_base64;
        if (!base64) {
          throw new Error('未返回图片数据');
        }
        const dataUrl = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
        setImageUrl(dataUrl);
        setStatus('生成完成，可添加到画布');
        setSnackbar({ open: true, message: '生成完成！', severity: 'success' });
      } else if (mode === 'layered') {
        setStatus('正在分层生成...');
        const res = await api.post('/generate-layered', {
          prompt: positive,
          aspect_ratio: aspectRatio,
          count: 4,
        });
        const rawLayers: LayerItem[] = res.data?.layers || [];
        if (!rawLayers.length) {
          throw new Error('未返回分层结果');
        }
        setLayers(rawLayers);
        setStatus('分层生成完成，可添加到画布');
        setSnackbar({ open: true, message: '分层生成完成！', severity: 'success' });
      } else if (mode === 'compose') {
        setStatus('正在组装素材...');
        // 组装模式：复用分层生成的思路，这里用 generate-layered 结果再布局一次
        const res = await api.post('/generate-layered', {
          prompt: positive,
          aspect_ratio: aspectRatio,
          count: 3,
        });
        const rawLayers: LayerItem[] = res.data?.layers || [];
        if (!rawLayers.length) {
          throw new Error('未返回组装结果');
        }
        setLayers(rawLayers);
        setStatus('组装完成，可添加到画布');
        setSnackbar({ open: true, message: '组装完成！', severity: 'success' });
      }
      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: `已基于「${effectivePrompt}」生成提示词和表单，你可以调整选项或直接生成。` },
      ]);
    } catch (error) {
      let message = '生成失败，请稍后重试。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data?.error || error.response.data?.message || message;
      }
      setStatus('');
      setSnackbar({ open: true, message, severity: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    const value = prompt.trim();
    if (!value) {
      setSnackbar({ open: true, message: '请输入你的场景想法。', severity: 'error' });
      return;
    }
    setMessages((prev) => [...prev, { role: 'user', text: value }]);
    await handleGenerate(value);
  };

  const handleAddToCanvas = useCallback(async () => {
    if (!imageUrl) return;
    try {
      const blob = await (await fetch(imageUrl)).blob();
      if (!blob.type.startsWith('image/')) {
        throw new Error('图片数据无效。');
      }
      const dataUrl = await blobToDataUrl(blob);
      const pendingRaw = sessionStorage.getItem('pendingCanvasItems');
      let pendingItems: { dataUrl: string; name: string }[] = [];
      if (pendingRaw) {
        try {
          pendingItems = JSON.parse(pendingRaw);
        } catch {
          pendingItems = [];
        }
      }
      pendingItems.push({ dataUrl, name: `Scene-${Date.now()}.png` });
      sessionStorage.setItem('pendingCanvasItems', JSON.stringify(pendingItems));
      setSnackbar({ open: true, message: '已添加到画布，前往“画布”页查看。', severity: 'success' });
    } catch (error) {
      const message = error instanceof Error ? error.message : '添加到画布失败。';
      setSnackbar({ open: true, message, severity: 'error' });
    }
  }, [imageUrl]);

  const handleAddLayersToCanvas = useCallback(() => {
    if (!layers.length) return;
    const pendingRaw = sessionStorage.getItem('pendingCanvasItems');
    let pendingItems: any[] = [];
    if (pendingRaw) {
      try {
        pendingItems = JSON.parse(pendingRaw);
      } catch {
        pendingItems = [];
      }
    }
    layers.forEach((layer) => {
      const dataUrl = layer.image_base64.startsWith('data:')
        ? layer.image_base64
        : `data:image/png;base64,${layer.image_base64}`;
      pendingItems.push({
        dataUrl,
        name: layer.name || layer.id,
        x: layer.placement?.x,
        y: layer.placement?.y,
        width: layer.width,
        height: layer.height,
        zIndex: layer.placement?.z_index,
      });
    });
    sessionStorage.setItem('pendingCanvasItems', JSON.stringify(pendingItems));
    setSnackbar({ open: true, message: '已添加到画布，前往“画布”页查看。', severity: 'success' });
  }, [layers]);

  return (
    <Container maxWidth={false} sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        AI 场景生成（M1/M2）
      </Typography>
      <ToggleButtonGroup
        value={mode}
        exclusive
        onChange={(_, value) => value && setMode(value)}
        sx={{ mb: 2 }}
      >
        <ToggleButton value="single">单图生成</ToggleButton>
        <ToggleButton value="layered">分层生成（M2）</ToggleButton>
        <ToggleButton value="compose">组装生成（M2）</ToggleButton>
      </ToggleButtonGroup>

      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                1. 描述你的想法（聊天 + 表单）
              </Typography>
              <Paper
                variant="outlined"
                sx={{
                  maxHeight: 260,
                  overflowY: 'auto',
                  p: 1.5,
                  mb: 2,
                  backgroundColor: '#f9fafb',
                }}
              >
                <Stack spacing={1.2}>
                  {messages.map((msg, idx) => (
                    <Box
                      key={`${msg.role}-${idx}`}
                      sx={{
                        display: 'flex',
                        justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                      }}
                    >
                      <Box
                        sx={{
                          px: 1.5,
                          py: 1,
                          maxWidth: '80%',
                          borderRadius: 2,
                          backgroundColor: msg.role === 'user' ? 'primary.light' : 'grey.100',
                          color: msg.role === 'user' ? 'primary.contrastText' : 'text.primary',
                        }}
                      >
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                          {msg.text}
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Stack>
              </Paper>
              <TextField
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="例如：一座未来城市的夜景，霓虹与云层交织"
                fullWidth
                multiline
                minRows={4}
              />
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel id="aspect-label">画幅</InputLabel>
                <Select
                  labelId="aspect-label"
                  label="画幅"
                  value={aspectRatio}
                  onChange={(e) => setAspectRatio(e.target.value)}
                  size="small"
                >
                  <MenuItem value="3:4">3:4</MenuItem>
                  <MenuItem value="1:1">1:1</MenuItem>
                  <MenuItem value="16:9">16:9</MenuItem>
                </Select>
              </FormControl>
              <Button
                variant="contained"
                color="primary"
                onClick={handleSend}
                disabled={isLoading}
                sx={{ mt: 2 }}
                fullWidth
              >
                {isLoading ? '生成中...' : '发送并生成'}
              </Button>
              {status && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    {status}
                  </Typography>
                  {isLoading && <LinearProgress sx={{ mt: 1 }} />}
                </Box>
              )}
              {positivePrompt && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    正向提示词
                  </Typography>
                  <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                    {positivePrompt}
                  </Typography>
                </Box>
              )}
              {layoutTips.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    表单建议
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    {layoutTips.map((tip) => (
                      <Chip key={tip} label={tip} size="small" />
                    ))}
                  </Stack>
                </Box>
              )}
              {formItems.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    快速选项
                  </Typography>
                  <Stack spacing={1.5}>
                    {formItems.map((item) => (
                      <Box key={item.title}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {item.title}
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                          {item.options.map((opt) => {
                            const selected = selectedOptions[item.title]?.includes(opt);
                            return (
                              <Chip
                                key={opt}
                                label={opt}
                                clickable
                                color={selected ? 'primary' : 'default'}
                                variant={selected ? 'filled' : 'outlined'}
                                onClick={() => {
                                  setSelectedOptions((prev) => {
                                    const current = prev[item.title] || [];
                                    const next = selected
                                      ? current.filter((v) => v !== opt)
                                      : [...current, opt];
                                    return { ...prev, [item.title]: next };
                                  });
                                }}
                              />
                            );
                          })}
                        </Stack>
                      </Box>
                    ))}
                  </Stack>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                2. 预览与落地
              </Typography>
              <Box
                sx={{
                  flexGrow: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: '#F8F9FA',
                  borderRadius: 1,
                  minHeight: 360,
                }}
              >
                {mode === 'single' ? (
                  imageUrl ? (
                    <img
                      src={imageUrl}
                      alt="生成结果"
                      style={{ maxWidth: '100%', maxHeight: '60vh', borderRadius: 8 }}
                    />
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      生成的图片会显示在这里
                    </Typography>
                  )
                ) : layers.length ? (
                  <Stack spacing={2} sx={{ width: '100%', maxHeight: '60vh', overflowY: 'auto' }}>
                    {layers.map((layer) => {
                      const src = layer.image_base64.startsWith('data:')
                        ? layer.image_base64
                        : `data:image/png;base64,${layer.image_base64}`;
                      return (
                        <Card key={layer.id} variant="outlined">
                          <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                              {layer.name}（{layer.layer_type}）
                            </Typography>
                            <img
                              src={src}
                              alt={layer.name}
                              style={{ maxWidth: '100%', borderRadius: 6 }}
                            />
                            {layer.placement && (
                              <Typography variant="caption" color="text.secondary">
                                位置: ({layer.placement.x}, {layer.placement.y}) z: {layer.placement.z_index}
                              </Typography>
                            )}
                          </CardContent>
                        </Card>
                      );
                    })}
                  </Stack>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    生成的图片/图层会显示在这里
                  </Typography>
                )}
              </Box>
              <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                {mode === 'single' ? (
                  <>
                    <Button
                      variant="contained"
                      onClick={handleAddToCanvas}
                      disabled={!imageUrl}
                    >
                      添加到画布
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() => {
                        if (imageUrl) {
                          const link = document.createElement('a');
                          link.href = imageUrl;
                          link.download = `scene-${Date.now()}.png`;
                          document.body.appendChild(link);
                          link.click();
                          document.body.removeChild(link);
                        }
                      }}
                      disabled={!imageUrl}
                    >
                      下载 PNG
                    </Button>
                  </>
                ) : (
                  <Button
                    variant="contained"
                    onClick={handleAddLayersToCanvas}
                    disabled={!layers.length}
                  >
                    添加所有图层到画布
                  </Button>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      {snackbar && (
        <Snackbar
          open={snackbar.open}
          autoHideDuration={5000}
          onClose={() => setSnackbar(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={() => setSnackbar(null)} severity={snackbar.severity} sx={{ width: '100%' }}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      )}
    </Container>
  );
};

export default SceneGenerationPage;

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error('读取图片数据失败。'));
      }
    };
    reader.onerror = () => reject(new Error('转换图片数据失败。'));
    reader.readAsDataURL(blob);
  });
}
