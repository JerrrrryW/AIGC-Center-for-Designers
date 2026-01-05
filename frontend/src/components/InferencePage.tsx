import React, { useCallback } from 'react';
import { Container, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import InferencePanel from './inference/InferencePanel';

const InferencePage: React.FC = () => {
  const navigate = useNavigate();

  const handleAddToCanvas = useCallback(
    async (imageUrl: string) => {
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`获取生成图片失败：${response.status}`);
      }
      const blob = await response.blob();
      if (!blob.type.startsWith('image/')) {
        throw new Error('生成内容解析失败。');
      }

      const dataUrl = await blobToDataUrl(blob);
      const pendingRaw = sessionStorage.getItem('pendingCanvasItems');
      let pendingItems: { dataUrl: string; name: string }[] = [];
      if (pendingRaw) {
        try {
          pendingItems = JSON.parse(pendingRaw);
        } catch (error) {
          console.error('解析待导入画布队列失败，已重置。', error);
          pendingItems = [];
        }
      }
      pendingItems.push({
        dataUrl,
        name: `生成-${Date.now()}.png`,
      });
      sessionStorage.setItem('pendingCanvasItems', JSON.stringify(pendingItems));
      navigate('/canvas');
    },
    [navigate],
  );

  return (
    <Container maxWidth={false}>
      <Typography variant="h4" component="h1" gutterBottom>
        Stable Diffusion 推理
      </Typography>
      <InferencePanel onAddToCanvas={handleAddToCanvas} />
    </Container>
  );
};

export default InferencePage;

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
