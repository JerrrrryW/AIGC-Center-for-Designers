import React, { useCallback, useMemo, useRef, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Grid,
  IconButton,
  Paper,
  Stack,
  TextField,
  Typography,
  Chip,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { api } from '../../api';

type ChatMessage = { role: 'user' | 'bot'; text: string; form?: FormSection[]; typing?: boolean };
type FormSection = { title: string; options: string[] };

type LayoutPayload = {
  layout_config?: any;
  layoutConfig?: any;
  text_response?: string;
  textResponse?: string;
};

const SceneChatPage: React.FC = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'bot', text: '你好！告诉我你想生成什么场景，我会先用选择题帮你把需求变清晰。' },
  ]);
  const [input, setInput] = useState('');
  const [selectedOptions, setSelectedOptions] = useState<Record<string, string[]>>({});
  const [isSending, setIsSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const latestForm = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const msg = messages[i];
      if (msg.role === 'bot' && msg.form?.length) return msg.form;
    }
    return null;
  }, [messages]);

  const isFormComplete = useMemo(() => {
    if (!latestForm?.length) return true;
    return latestForm.every((section) => (selectedOptions[section.title] ?? []).length > 0);
  }, [latestForm, selectedOptions]);

  const chatHistory = useMemo(() => {
    return messages
      .filter((m) => !m.typing)
      .map((m) => `${m.role === 'user' ? 'User' : 'Bot'}: ${m.text}`)
      .join('\n');
  }, [messages]);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, []);

  const toggleOption = useCallback((title: string, option: string) => {
    setSelectedOptions((prev) => {
      const current = prev[title] ?? [];
      const next = current.includes(option)
        ? current.filter((v) => v !== option)
        : [...current, option];
      return { ...prev, [title]: next };
    });
  }, []);

  const handleSend = useCallback(async () => {
    const value = input.trim();
    if (!value || isSending) return;
    setIsSending(true);
    setInput('');
    const nextMessages = [...messages, { role: 'user' as const, text: value }];
    setMessages([...nextMessages, { role: 'bot', text: 'Agent 输入中...', typing: true }]);
    queueMicrotask(scrollToBottom);

    try {
      const res = await api.post('/chat', {
        messages: nextMessages.map((m) => ({
          role: m.role === 'bot' ? 'assistant' : 'user',
          content: m.text,
        })),
        selected_options: selectedOptions,
      });
      const obj = res.data?.json_object ?? res.data;
      const botText = obj?.text_response ?? '好的，我们继续。';
      const form = normalizeForm(obj?.form);
      setMessages((prev) => {
        const withoutTyping = prev.filter((m) => !m.typing);
        return [...withoutTyping, { role: 'bot', text: botText, form }];
      });
    } catch (error) {
      let message = '连接服务器失败，请检查后端服务。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data?.error || error.response.data?.message || message;
      }
      setMessages((prev) => {
        const withoutTyping = prev.filter((m) => !m.typing);
        return [...withoutTyping, { role: 'bot', text: message }];
      });
    } finally {
      setIsSending(false);
      queueMicrotask(scrollToBottom);
    }
  }, [input, isSending, messages, scrollToBottom, selectedOptions]);

  const handleGenerateEdit = useCallback(async () => {
    if (!isFormComplete || isSending) return;
    const initialPrompt =
      messages.find((m) => m.role === 'user')?.text ?? '';

    try {
      const [layoutRes, layerPlanRes] = await Promise.all([
        api.post('/generate-layout', {
          prompt: initialPrompt,
          chat_history: chatHistory,
          selected_options: selectedOptions,
        }),
        api.post('/analyze-layer-plan', {
          chat_history: chatHistory,
          selected_options: selectedOptions,
          ui_state: {},
        }),
      ]);

      const layoutPayload: LayoutPayload = layoutRes.data;
      navigate('/scene/edit', {
        state: {
          layoutConfig: layoutPayload.layout_config ?? layoutPayload.layoutConfig,
          generatedPrompt: layoutPayload.text_response ?? layoutPayload.textResponse ?? '',
          chatHistory,
          selectedOptions,
          layerPlan: layerPlanRes.data,
        },
      });
    } catch (error) {
      let message = '生成编辑界面失败。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data?.error || error.response.data?.message || message;
      }
      setMessages((prev) => [...prev, { role: 'bot', text: message }]);
    }
  }, [chatHistory, isFormComplete, isSending, messages, navigate, selectedOptions]);

  return (
    <Container maxWidth={false} sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        AI 场景生成 · Chat
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                对话
              </Typography>
              <Paper
                ref={scrollRef}
                variant="outlined"
                sx={{
                  flexGrow: 1,
                  minHeight: 320,
                  maxHeight: { xs: 360, md: 520 },
                  overflowY: 'auto',
                  p: 1.5,
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
                          maxWidth: '84%',
                          borderRadius: 2,
                          backgroundColor: msg.role === 'user' ? 'primary.main' : 'grey.100',
                          color: msg.role === 'user' ? 'primary.contrastText' : 'text.primary',
                        }}
                      >
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                          {msg.text}
                        </Typography>
                        {!msg.typing && msg.form?.length ? (
                          <Box sx={{ mt: 1 }}>
                            <Card variant="outlined" sx={{ bgcolor: 'background.paper' }}>
                              <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                                <Typography variant="subtitle2" gutterBottom>
                                  选择题表单
                                </Typography>
                                <Stack spacing={1.5}>
                                  {normalizeForm(msg.form).map((section, sectionIndex) => (
                                    <Box key={`${section.title}-${sectionIndex}`}>
                                      <Typography variant="caption" sx={{ opacity: 0.9, display: 'block' }}>
                                        {section.title}
                                      </Typography>
                                      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
                                        {(Array.isArray(section.options) ? section.options : []).map((opt) => {
                                          const selected = selectedOptions[section.title]?.includes(opt);
                                          return (
                                            <Chip
                                              key={opt}
                                              label={opt}
                                              size="small"
                                              clickable
                                              color={selected ? 'primary' : 'default'}
                                              variant={selected ? 'filled' : 'outlined'}
                                              onClick={() => toggleOption(section.title, opt)}
                                            />
                                          );
                                        })}
                                      </Stack>
                                    </Box>
                                  ))}
                                  {msg.form.length > 0 && !isFormComplete ? (
                                    <Typography variant="caption" color="error">
                                      请为每项至少选择一个选项，才能进入编辑界面。
                                    </Typography>
                                  ) : null}
                                </Stack>
                              </CardContent>
                            </Card>
                          </Box>
                        ) : null}
                      </Box>
                    </Box>
                  ))}
                </Stack>
              </Paper>

              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <TextField
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="请输入你想生成的内容"
                  fullWidth
                  multiline
                  minRows={2}
                  onKeyDown={(e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                />
                <IconButton color="primary" onClick={handleSend} disabled={isSending}>
                  <SendIcon />
                </IconButton>
              </Box>
              <Button
                variant="contained"
                sx={{ mt: 2 }}
                onClick={handleGenerateEdit}
                disabled={isSending || (latestForm ? !isFormComplete : false)}
              >
                生成编辑界面
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                平板模式参考
              </Typography>
              <Typography variant="body2" color="text.secondary">
                LiteDraw 在平板横屏下会把“聊天”和“表单/预览”分栏。这里保留右侧面板作为辅助信息；
                表单本身会作为卡片出现在对话中，支持多轮对话持续更新。
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  当前选择
                </Typography>
                {Object.keys(selectedOptions).length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    暂无
                  </Typography>
                ) : (
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    {Object.entries(selectedOptions).flatMap(([k, values]) =>
                      values.map((v) => <Chip key={`${k}:${v}`} label={`${k}: ${v}`} size="small" />),
                    )}
                  </Stack>
                )}
              </Box>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  对话提示
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  你可以继续补充细节（多轮对话），或选完表单后进入编辑界面准备素材与预览。
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default SceneChatPage;

function normalizeForm(input: unknown): FormSection[] {
  if (!Array.isArray(input)) {
    return [];
  }
  const sections: FormSection[] = [];
  input.forEach((raw, index) => {
    if (!raw || typeof raw !== 'object') {
      return;
    }
    const maybeTitle = (raw as any).title;
    const title = typeof maybeTitle === 'string' && maybeTitle.trim() ? maybeTitle.trim() : `问题 ${index + 1}`;
    const maybeOptions = (raw as any).options;
    let options: string[] = [];
    if (Array.isArray(maybeOptions)) {
      options = maybeOptions.map((v) => String(v)).map((v) => v.trim()).filter(Boolean);
    } else if (typeof maybeOptions === 'string') {
      options = maybeOptions
        .split(/[,，、;\n]+/g)
        .map((v) => v.trim())
        .filter(Boolean);
    }
    options = Array.from(new Set(options)).slice(0, 6);
    if (options.length === 0) {
      return;
    }
    sections.push({ title, options });
  });
  return sections;
}
