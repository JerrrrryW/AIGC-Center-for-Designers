import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Tooltip,
  Divider,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import FlipToFrontIcon from '@mui/icons-material/FlipToFront';
import FlipToBackIcon from '@mui/icons-material/FlipToBack';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';
import DownloadIcon from '@mui/icons-material/Download';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import CloseIcon from '@mui/icons-material/Close';
import InferencePanel from './inference/InferencePanel';

type BaseCanvasItem = {
  id: string;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
  kind: 'image' | 'text' | 'shape';
};

type ImageCanvasItem = BaseCanvasItem & {
  kind: 'image';
  src: string;
  naturalWidth: number;
  naturalHeight: number;
};

type TextCanvasItem = BaseCanvasItem & {
  kind: 'text';
  text: string;
  style?: {
    color?: string;
    fontSize?: number;
    fontWeight?: number;
    align?: 'left' | 'center' | 'right';
    backgroundColor?: string;
    padding?: number;
    lineHeight?: number;
  };
};

type ShapeCanvasItem = BaseCanvasItem & {
  kind: 'shape';
  style?: {
    fill?: string;
    radius?: number;
  };
};

type CanvasItem = ImageCanvasItem | TextCanvasItem | ShapeCanvasItem;

type PendingCanvasItem = {
  kind?: 'image' | 'text' | 'shape';
  dataUrl?: string;
  name?: string;
  text?: string;
  style?: any;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  zIndex?: number;
};

const CANVAS_DIMENSION = 4000;
const MIN_ITEM_SIZE = 48;
const MAX_ZOOM = 2;
const MIN_ZOOM = 0.25;
const ZOOM_STEP = 0.1;

const CanvasPage: React.FC = () => {
  const [items, setItems] = useState<CanvasItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [isExporting, setIsExporting] = useState(false);
  const [isInferenceOpen, setIsInferenceOpen] = useState(false);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragState = useRef<{ id: string; offsetX: number; offsetY: number } | null>(null);
  const resizeState = useRef<{
    id: string;
    originX: number;
    originY: number;
    startWidth: number;
    startHeight: number;
  } | null>(null);
  const createdUrls = useRef<Set<string>>(new Set());
  const zoomRef = useRef(zoom);

  useEffect(() => {
    zoomRef.current = zoom;
  }, [zoom]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.scrollLeft = (CANVAS_DIMENSION - container.clientWidth) / 2;
    container.scrollTop = (CANVAS_DIMENSION - container.clientHeight) / 2;
  }, []);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!canvasRef.current) return;
      const canvasBounds = canvasRef.current.getBoundingClientRect();
      const pointerX = (event.clientX - canvasBounds.left) / zoomRef.current;
      const pointerY = (event.clientY - canvasBounds.top) / zoomRef.current;

      if (dragState.current) {
        const { id, offsetX, offsetY } = dragState.current;
        setItems((prev) =>
          prev.map((item) => {
            if (item.id !== id) return item;
            const newX = clamp(pointerX - offsetX, 0, CANVAS_DIMENSION - item.width);
            const newY = clamp(pointerY - offsetY, 0, CANVAS_DIMENSION - item.height);
            return { ...item, x: newX, y: newY };
          }),
        );
      } else if (resizeState.current) {
        const { id, originX, originY, startWidth, startHeight } = resizeState.current;
        setItems((prev) =>
          prev.map((item) => {
            if (item.id !== id) return item;
            const nextWidth = clamp(
              startWidth + (pointerX - originX),
              MIN_ITEM_SIZE,
              CANVAS_DIMENSION - item.x,
            );
            const nextHeight = clamp(
              startHeight + (pointerY - originY),
              MIN_ITEM_SIZE,
              CANVAS_DIMENSION - item.y,
            );
            return { ...item, width: nextWidth, height: nextHeight };
          }),
        );
      }
    };

    const handleMouseUp = () => {
      dragState.current = null;
      resizeState.current = null;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  useEffect(() => {
    return () => {
      createdUrls.current.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  const sortedItems = useMemo(() => [...items].sort((a, b) => a.zIndex - b.zIndex), [items]);

  const addCanvasItem = useCallback(
    (item: Omit<CanvasItem, 'id' | 'zIndex'> & { zIndex?: number }) => {
      let created: CanvasItem | null = null;
      setItems((prev) => {
        const maxZ = prev.reduce((acc, current) => Math.max(acc, current.zIndex), 0);
        created = {
          ...item,
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          zIndex: item.zIndex ?? maxZ + 1,
        } as CanvasItem;
        return [...prev, created as CanvasItem];
      });
      if (created) setSelectedId(created.id);
    },
    [],
  );

  const addCanvasImageItem = useCallback(
    (
      src: string,
      name: string,
      naturalWidth: number,
      naturalHeight: number,
      preset?: { x?: number; y?: number; width?: number; height?: number; zIndex?: number },
    ) => {
      if (!naturalWidth || !naturalHeight) return;
      const baseWidth = preset?.width ?? Math.min(naturalWidth, 320);
      const scale = baseWidth / naturalWidth;
      const baseHeight = preset?.height ?? naturalHeight * scale;
      const center = CANVAS_DIMENSION / 2;
      addCanvasItem({
        kind: 'image',
        src,
        name,
        x: preset?.x ?? center - baseWidth / 2,
        y: preset?.y ?? center - baseHeight / 2,
        width: baseWidth,
        height: baseHeight,
        naturalWidth,
        naturalHeight,
        zIndex: preset?.zIndex,
      });
    },
    [addCanvasItem],
  );

  const handleBackgroundMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === contentRef.current) setSelectedId(null);
  };

  const handleItemMouseDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>, id: string) => {
      event.stopPropagation();
      event.preventDefault();
      if (!canvasRef.current) return;
      const item = items.find((entry) => entry.id === id);
      if (!item) return;
      const bounds = canvasRef.current.getBoundingClientRect();
      const pointerX = (event.clientX - bounds.left) / zoomRef.current;
      const pointerY = (event.clientY - bounds.top) / zoomRef.current;
      dragState.current = { id, offsetX: pointerX - item.x, offsetY: pointerY - item.y };
      setSelectedId(id);
    },
    [items],
  );

  const handleResizeMouseDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>, id: string) => {
      event.stopPropagation();
      event.preventDefault();
      if (!canvasRef.current) return;
      const item = items.find((entry) => entry.id === id);
      if (!item) return;
      const bounds = canvasRef.current.getBoundingClientRect();
      const pointerX = (event.clientX - bounds.left) / zoomRef.current;
      const pointerY = (event.clientY - bounds.top) / zoomRef.current;
      resizeState.current = {
        id,
        originX: pointerX,
        originY: pointerY,
        startWidth: item.width,
        startHeight: item.height,
      };
      setSelectedId(id);
    },
    [items],
  );

  const triggerFileUpload = () => fileInputRef.current?.click();

  const handleFilesSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;

    for (const file of files) {
      if (!file.type.startsWith('image/')) continue;
      const src = URL.createObjectURL(file);
      try {
        const image = await loadImageElement(src);
        createdUrls.current.add(src);
        addCanvasImageItem(
          src,
          file.name,
          image.naturalWidth || image.width,
          image.naturalHeight || image.height,
        );
      } catch {
        URL.revokeObjectURL(src);
      }
    }
    event.target.value = '';
  };

  const handleAddGeneratedImage = useCallback(
    async (imageUrl: string) => {
      let objectUrl: string | null = null;
      try {
        const response = await fetch(imageUrl);
        if (!response.ok) throw new Error(`获取生成图片失败：${response.status}`);
        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);
        const image = await loadImageElement(objectUrl);
        createdUrls.current.add(objectUrl);
        addCanvasImageItem(
          objectUrl,
          `生成-${Date.now()}.png`,
          image.naturalWidth || image.width,
          image.naturalHeight || image.height,
        );
      } catch (error) {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        console.error('添加生成图片到画布失败:', error);
        throw error;
      }
    },
    [addCanvasImageItem],
  );

  useEffect(() => {
    const pendingRaw = sessionStorage.getItem('pendingCanvasItems');
    if (!pendingRaw) return;
    sessionStorage.removeItem('pendingCanvasItems');

    let pendingItems: PendingCanvasItem[] = [];
    try {
      pendingItems = JSON.parse(pendingRaw);
    } catch (error) {
      console.error('解析待导入画布素材失败:', error);
      return;
    }

    const importPending = async () => {
      for (const item of pendingItems) {
        try {
          if ((item.kind || 'image') === 'image' && item.dataUrl) {
            const blob = dataUrlToBlob(item.dataUrl);
            const objectUrl = URL.createObjectURL(blob);
            const image = await loadImageElement(objectUrl);
            createdUrls.current.add(objectUrl);
            addCanvasImageItem(
              objectUrl,
              item.name || `生成-${Date.now()}.png`,
              image.naturalWidth || image.width,
              image.naturalHeight || image.height,
              {
                x: item.x,
                y: item.y,
                width: item.width,
                height: item.height,
                zIndex: item.zIndex,
              },
            );
            continue;
          }

          if (item.kind === 'text') {
            addCanvasItem({
              kind: 'text',
              name: item.name || '文字',
              text: item.text || '',
              x: item.x ?? 400,
              y: item.y ?? 400,
              width: item.width ?? 480,
              height: item.height ?? 160,
              style: item.style,
              zIndex: item.zIndex,
            });
            continue;
          }

          if (item.kind === 'shape') {
            addCanvasItem({
              kind: 'shape',
              name: item.name || '色块',
              x: item.x ?? 400,
              y: item.y ?? 400,
              width: item.width ?? 320,
              height: item.height ?? 200,
              style: item.style,
              zIndex: item.zIndex,
            });
          }
        } catch (error) {
          console.error('导入已保存的画布素材失败:', error);
        }
      }
    };

    importPending();
  }, [addCanvasImageItem, addCanvasItem]);

  const handleDeleteSelected = () => {
    if (!selectedId) return;
    setItems((prev) => {
      const target = prev.find((item) => item.id === selectedId);
      if (target?.kind === 'image' && target.src.startsWith('blob:')) {
        createdUrls.current.delete(target.src);
        URL.revokeObjectURL(target.src);
      }
      return prev.filter((item) => item.id !== selectedId);
    });
    setSelectedId(null);
  };

  const handleBringForward = () => {
    if (!selectedId) return;
    setItems((prev) => {
      const maxZ = prev.reduce((acc, item) => Math.max(acc, item.zIndex), 0);
      return prev.map((item) => (item.id === selectedId ? { ...item, zIndex: maxZ + 1 } : item));
    });
  };

  const handleSendBackward = () => {
    if (!selectedId) return;
    setItems((prev) => {
      const minZ = prev.reduce((acc, item) => Math.min(acc, item.zIndex), Infinity);
      return prev.map((item) => (item.id === selectedId ? { ...item, zIndex: minZ - 1 } : item));
    });
  };

  const handleZoomIn = () => setZoom((prev) => Math.min(MAX_ZOOM, roundToTwo(prev + ZOOM_STEP)));
  const handleZoomOut = () => setZoom((prev) => Math.max(MIN_ZOOM, roundToTwo(prev - ZOOM_STEP)));

  const handleResetView = () => {
    setZoom(1);
    const container = containerRef.current;
    if (!container) return;
    container.scrollLeft = (CANVAS_DIMENSION - container.clientWidth) / 2;
    container.scrollTop = (CANVAS_DIMENSION - container.clientHeight) / 2;
  };

  const handleLayerSelect = (id: string) => {
    setSelectedId(id);
    const item = items.find((entry) => entry.id === id);
    const container = containerRef.current;
    if (!item || !container) return;
    const viewWidth = container.clientWidth / zoom;
    const viewHeight = container.clientHeight / zoom;
    container.scrollLeft = (item.x + item.width / 2) * zoom - viewWidth / 2;
    container.scrollTop = (item.y + item.height / 2) * zoom - viewHeight / 2;
  };

  const handleExport = useCallback(async () => {
    if (!items.length || isExporting) return;
    setIsExporting(true);
    try {
      const minX = Math.min(...items.map((item) => item.x));
      const minY = Math.min(...items.map((item) => item.y));
      const maxX = Math.max(...items.map((item) => item.x + item.width));
      const maxY = Math.max(...items.map((item) => item.y + item.height));
      const padding = 32;
      const exportWidth = Math.ceil(maxX - minX + padding * 2);
      const exportHeight = Math.ceil(maxY - minY + padding * 2);
      if (exportWidth <= 0 || exportHeight <= 0) return;

      const canvasElement = document.createElement('canvas');
      canvasElement.width = exportWidth;
      canvasElement.height = exportHeight;
      const context = canvasElement.getContext('2d');
      if (!context) throw new Error('无法创建画布上下文。');
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, exportWidth, exportHeight);

      const orderedItems = [...items].sort((a, b) => a.zIndex - b.zIndex);
      for (const item of orderedItems) {
        const offsetX = padding + item.x - minX;
        const offsetY = padding + item.y - minY;
        if (item.kind === 'image') {
          try {
            const image = await loadImageElement(item.src);
            const naturalWidth = item.naturalWidth || image.naturalWidth || image.width;
            const naturalHeight = item.naturalHeight || image.naturalHeight || image.height;
            const scale = Math.min(item.width / naturalWidth, item.height / naturalHeight);
            const drawWidth = naturalWidth * scale;
            const drawHeight = naturalHeight * scale;
            context.drawImage(
              image,
              offsetX + (item.width - drawWidth) / 2,
              offsetY + (item.height - drawHeight) / 2,
              drawWidth,
              drawHeight,
            );
          } catch {
            // ignore failed image layer
          }
          continue;
        }
        if (item.kind === 'shape') {
          drawRoundedRect(
            context,
            offsetX,
            offsetY,
            item.width,
            item.height,
            item.style?.radius ?? 24,
            item.style?.fill ?? 'rgba(255,255,255,0.72)',
          );
          continue;
        }
        drawTextLayer(context, item, offsetX, offsetY);
      }

      const blob = await canvasToBlob(canvasElement);
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = `canvas-export-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
    } finally {
      setIsExporting(false);
    }
  }, [isExporting, items]);

  const selectedItem = items.find((item) => item.id === selectedId) || null;

  return (
    <Box sx={{ position: 'relative', height: 'calc(100vh - 112px)', width: '100%' }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
        <Box>
          <Typography variant="h4" component="h1">
            画布
          </Typography>
          <Typography variant="body2" color="text.secondary">
            导入背景、素材、文字和色块，拖拽排版并导出 PNG。
          </Typography>
        </Box>
      </Stack>

      <Box
        ref={containerRef}
        sx={{
          flex: 1,
          width: '100%',
          height: '100%',
          borderRadius: 2,
          overflow: 'auto',
          position: 'relative',
          background:
            'radial-gradient(circle at center, rgba(255,255,255,0.4) 0%, rgba(240,242,245,0.9) 100%)',
          boxShadow: 'inset 0 0 0 1px rgba(33,37,41,0.04)',
        }}
      >
        <Box
          ref={canvasRef}
          sx={{
            position: 'relative',
            width: CANVAS_DIMENSION,
            height: CANVAS_DIMENSION,
            margin: '0 auto',
            backgroundImage:
              'linear-gradient(0deg, rgba(0,0,0,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.04) 1px, transparent 1px)',
            backgroundSize: '64px 64px',
            backgroundColor: '#f6f7f9',
          }}
        >
          <Box
            ref={contentRef}
            onMouseDown={handleBackgroundMouseDown}
            sx={{
              position: 'absolute',
              inset: 0,
              transform: `scale(${zoom})`,
              transformOrigin: 'top left',
              pointerEvents: 'auto',
            }}
          >
            {sortedItems.map((item) => (
              <Box
                key={item.id}
                onMouseDown={(event) => handleItemMouseDown(event, item.id)}
                sx={{
                  position: 'absolute',
                  left: item.x,
                  top: item.y,
                  width: item.width,
                  height: item.height,
                  cursor: 'move',
                  zIndex: item.zIndex,
                  border: item.id === selectedId ? '2px solid #007BFF' : '1px solid rgba(33, 37, 41, 0.12)',
                  borderRadius: 1,
                  boxShadow:
                    item.id === selectedId
                      ? '0 0 0 4px rgba(0, 123, 255, 0.16)'
                      : '0 4px 12px rgba(15,23,42,0.12)',
                  backgroundColor: item.kind === 'image' ? '#fff' : 'transparent',
                  overflow: 'hidden',
                  transition: 'border 0.15s ease, box-shadow 0.15s ease',
                  userSelect: 'none',
                }}
              >
                {item.kind === 'image' ? (
                  <Box
                    component="img"
                    src={item.src}
                    alt={item.name}
                    sx={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain',
                      display: 'block',
                      pointerEvents: 'none',
                      userSelect: 'none',
                    }}
                    draggable={false}
                  />
                ) : item.kind === 'shape' ? (
                  <Box
                    sx={{
                      width: '100%',
                      height: '100%',
                      background: item.style?.fill || 'rgba(255,255,255,0.72)',
                      borderRadius: `${item.style?.radius ?? 24}px`,
                      pointerEvents: 'none',
                    }}
                  />
                ) : (
                  <Box
                    sx={{
                      width: '100%',
                      height: '100%',
                      background: item.style?.backgroundColor || 'transparent',
                      color: item.style?.color || '#111827',
                      p: `${item.style?.padding ?? 18}px`,
                      fontSize: `${Math.max(
                        12,
                        Math.round((item.style?.fontSize || 48) * Math.min(item.width / 320, item.height / 120)),
                      )}px`,
                      fontWeight: item.style?.fontWeight || 700,
                      lineHeight: item.style?.lineHeight || 1.2,
                      textAlign: item.style?.align || 'left',
                      display: 'flex',
                      alignItems: 'center',
                      whiteSpace: 'pre-wrap',
                      overflow: 'hidden',
                      pointerEvents: 'none',
                    }}
                  >
                    {item.text}
                  </Box>
                )}
                <Box
                  onMouseDown={(event) => handleResizeMouseDown(event, item.id)}
                  sx={{
                    position: 'absolute',
                    width: 16,
                    height: 16,
                    borderRadius: '4px',
                    bottom: 6,
                    right: 6,
                    backgroundColor: '#007BFF',
                    boxShadow: '0 2px 6px rgba(0, 123, 255, 0.45)',
                    cursor: 'nwse-resize',
                  }}
                />
              </Box>
            ))}
          </Box>
        </Box>
      </Box>

      <Paper
        elevation={6}
        sx={{
          position: 'fixed',
          left: { xs: 16, sm: 288 },
          right: 16,
          bottom: 24,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 1,
          padding: '12px 16px',
          borderRadius: 999,
          backdropFilter: 'blur(12px)',
          backgroundColor: 'rgba(255,255,255,0.88)',
          zIndex: 1200,
        }}
      >
        <Tooltip title="上传素材">
          <IconButton color="primary" onClick={triggerFileUpload}>
            <AddPhotoAlternateIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="AI 生图">
          <IconButton color="primary" onClick={() => setIsInferenceOpen(true)}>
            <AutoAwesomeIcon />
          </IconButton>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
        <Tooltip title="置于最上层">
          <span>
            <IconButton onClick={handleBringForward} disabled={!selectedId} color="primary">
              <FlipToFrontIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="置于最底层">
          <span>
            <IconButton onClick={handleSendBackward} disabled={!selectedId} color="primary">
              <FlipToBackIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="删除素材">
          <span>
            <IconButton onClick={handleDeleteSelected} disabled={!selectedId} color="secondary">
              <DeleteOutlineIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title={isExporting ? '导出中...' : '导出PNG'}>
          <span>
            <IconButton onClick={handleExport} disabled={isExporting || !items.length} color="primary">
              <DownloadIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
        <Tooltip title="缩小">
          <IconButton onClick={handleZoomOut}>
            <ZoomOutIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="放大">
          <IconButton onClick={handleZoomIn}>
            <ZoomInIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="回到中心">
          <IconButton onClick={handleResetView}>
            <CenterFocusStrongIcon />
          </IconButton>
        </Tooltip>
      </Paper>

      <Paper
        elevation={4}
        sx={{
          position: 'fixed',
          top: 120,
          right: 24,
          width: 260,
          maxHeight: '60vh',
          overflow: 'auto',
          borderRadius: 3,
          backgroundColor: 'rgba(255,255,255,0.92)',
          zIndex: 1100,
        }}
      >
        <Box sx={{ px: 2, py: 1.5 }}>
          <Typography variant="subtitle2" color="text.secondary">
            图层
          </Typography>
          {selectedItem ? (
            <Typography variant="caption" color="text.secondary">
              已选：{selectedItem.name} / {selectedItem.kind}
            </Typography>
          ) : null}
        </Box>
        <Divider />
        {sortedItems.length ? (
          <List dense disablePadding>
            {[...sortedItems].reverse().map((item, index) => (
              <ListItemButton
                key={item.id}
                selected={item.id === selectedId}
                onClick={() => handleLayerSelect(item.id)}
                sx={{ py: 1 }}
              >
                <ListItemText
                  primaryTypographyProps={{
                    noWrap: true,
                    fontSize: 13,
                    fontWeight: item.id === selectedId ? 600 : 500,
                  }}
                  primary={`${sortedItems.length - index}. ${item.name}`}
                  secondary={item.kind}
                />
              </ListItemButton>
            ))}
          </List>
        ) : (
          <Box sx={{ px: 2, py: 2 }}>
            <Typography variant="body2" color="text.secondary">
              暂无素材，点击下方工具栏上传。
            </Typography>
          </Box>
        )}
      </Paper>

      <Dialog open={isInferenceOpen} onClose={() => setIsInferenceOpen(false)} fullWidth maxWidth="lg">
        <DialogTitle
          sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', pr: 2 }}
        >
          AI 生图
          <IconButton onClick={() => setIsInferenceOpen(false)}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          <InferencePanel onAddToCanvas={handleAddGeneratedImage} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsInferenceOpen(false)}>关闭</Button>
        </DialogActions>
      </Dialog>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={handleFilesSelected}
      />
    </Box>
  );
};

function drawRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
  fill: string,
) {
  const r = Math.min(radius, width / 2, height / 2);
  context.save();
  context.fillStyle = fill;
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
  context.fill();
  context.restore();
}

function drawTextLayer(context: CanvasRenderingContext2D, item: TextCanvasItem, x: number, y: number) {
  const padding = item.style?.padding ?? 18;
  if (item.style?.backgroundColor) {
    drawRoundedRect(context, x, y, item.width, item.height, 20, item.style.backgroundColor);
  }
  const fontSize = item.style?.fontSize || 48;
  const fontWeight = item.style?.fontWeight || 700;
  const color = item.style?.color || '#111827';
  const lineHeight = (item.style?.lineHeight || 1.2) * fontSize;
  const maxTextWidth = Math.max(32, item.width - padding * 2);
  context.save();
  context.fillStyle = color;
  context.font = `${fontWeight} ${fontSize}px Arial, sans-serif`;
  context.textAlign =
    item.style?.align === 'center' ? 'center' : item.style?.align === 'right' ? 'right' : 'left';
  context.textBaseline = 'top';
  const lines = wrapText(context, item.text, maxTextWidth);
  const anchorX =
    item.style?.align === 'center'
      ? x + item.width / 2
      : item.style?.align === 'right'
        ? x + item.width - padding
        : x + padding;
  let cursorY = y + padding;
  for (const line of lines) {
    context.fillText(line, anchorX, cursorY, maxTextWidth);
    cursorY += lineHeight;
  }
  context.restore();
}

function wrapText(context: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const chars = Array.from(text || '');
  const lines: string[] = [];
  let current = '';
  chars.forEach((char) => {
    if (char === '\n') {
      if (current) lines.push(current);
      current = '';
      return;
    }
    const test = current + char;
    if (context.measureText(test).width > maxWidth && current) {
      lines.push(current);
      current = char;
    } else {
      current = test;
    }
  });
  if (current) lines.push(current);
  return lines.length ? lines : [''];
}

function clamp(value: number, min: number, max: number) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function roundToTwo(value: number) {
  return Math.round(value * 100) / 100;
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('导出失败'));
    }, 'image/png');
  });
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('加载图片失败。'));
    image.src = src;
  });
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [metadata, base64Data] = dataUrl.split(',');
  if (!metadata || !base64Data) throw new Error('无效的数据 URL。');
  const match = metadata.match(/data:(.*?);base64/);
  const mime = match?.[1] || 'image/png';
  const binary = atob(base64Data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

export default CanvasPage;
