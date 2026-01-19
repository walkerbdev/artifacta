import React, { useState, useEffect, useMemo, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import './ChatTab.scss';
import { apiClient } from '@/core/api/ApiClient';

export const ChatTab = ({ selectedRunIds, allRuns: _allRuns }) => {
  const [runData, setRunData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState('gpt-4o-mini');
  const [apiKey, setApiKey] = useState('');
  const [showSetup, setShowSetup] = useState(true);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [inputHeight, setInputHeight] = useState(200);
  const [isResizing, setIsResizing] = useState(false);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const userHasScrolledRef = useRef(false);

  // Load model and API key from localStorage
  useEffect(() => {
    const storedModel = localStorage.getItem('llm_model');
    const storedKey = localStorage.getItem('llm_api_key');
    if (storedModel && storedKey) {
      setModel(storedModel);
      setApiKey(storedKey);
      setShowSetup(false);
    }
  }, []);

  // Detect if user manually scrolls
  const handleScroll = () => {
    if (!messagesContainerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    // User has scrolled up if they're more than 150px from bottom
    userHasScrolledRef.current = distanceFromBottom > 150;
  };

  // Auto-scroll to bottom only if user hasn't manually scrolled up
  useEffect(() => {
    if (!userHasScrolledRef.current && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Handle resize of input area
  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e) => {
      e.preventDefault();
      const containerHeight = window.innerHeight;
      const newHeight = containerHeight - e.clientY;
      // No restrictions - resize freely
      setInputHeight(Math.max(50, newHeight));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.classList.remove('resizing-chat');
    };

    document.body.classList.add('resizing-chat');
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.classList.remove('resizing-chat');
    };
  }, [isResizing]);

  // Fetch run data when selectedRunIds changes
  useEffect(() => {
    if (!selectedRunIds || selectedRunIds.length === 0) {
      setRunData(null);
      setMessages([]);
      return;
    }

    const fetchRunData = async () => {
      setLoading(true);
      try {
        const runsPromises = selectedRunIds.map(id => apiClient.getRun(id));
        const runs = await Promise.all(runsPromises);

        const artifactsPromises = selectedRunIds.map(id =>
          apiClient.get(`/api/artifacts/${id}`)
        );
        const artifactsArrays = await Promise.all(artifactsPromises);

        const combined = runs.map((run, idx) => ({
          ...run,
          artifacts: artifactsArrays[idx] || []
        }));

        setRunData(combined);
        setMessages([]);
      } catch (error) {
        console.error('Failed to fetch run data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchRunData();
  }, [selectedRunIds]);

  // Build context from run data
  const systemMessage = useMemo(() => {
    if (!runData || runData.length === 0) {
      return 'You are a helpful ML experiment assistant.';
    }

    let context = 'You are an AI assistant analyzing ML experiment results.\n\n';

    runData.forEach((run, idx) => {
      context += `# Experiment ${idx + 1}: ${run.name || run.run_id}\n\n`;

      if (run.config && Object.keys(run.config).length > 0) {
        context += '## Configuration\n```json\n';
        context += JSON.stringify(run.config, null, 2);
        context += '\n```\n\n';
      }

      if (run.summary && Object.keys(run.summary).length > 0) {
        context += '## Final Metrics\n';
        Object.entries(run.summary).forEach(([key, value]) => {
          context += `- ${key}: ${value}\n`;
        });
        context += '\n';
      }

      if (run.structured_data && Object.keys(run.structured_data).length > 0) {
        context += '## Logged Data\n```json\n';
        context += JSON.stringify(run.structured_data, null, 2);
        context += '\n```\n\n';
      }

      // Include artifacts with file tree structure and size limits
      const MAX_ARTIFACT_CONTEXT = 100000; // ~100KB total for artifacts
      let totalArtifactSize = 0;

      const artifacts = run.artifacts?.filter(a => a.content) || [];
      if (artifacts.length > 0) {
        context += '## Artifacts\n\n';

        for (const artifact of artifacts) {
          try {
            const fileCollection = JSON.parse(artifact.content);
            const files = fileCollection.files || [];

            context += `### ${artifact.name}\n`;
            context += `- Files: ${fileCollection.total_files}\n`;
            context += `- Size: ${(fileCollection.total_size / 1024).toFixed(1)} KB\n\n`;

            // Show directory tree structure
            if (files.length > 1) {
              context += '**File structure:**\n```\n';
              files.forEach(f => context += `${f.path}\n`);
              context += '```\n\n';
            }

            // Include file contents with size limit
            const filesWithContent = files.filter(f => f.content);
            if (filesWithContent.length > 0) {
              context += '**File contents:**\n\n';

              for (const file of filesWithContent) {
                const contentSize = file.content.length;

                if (totalArtifactSize + contentSize > MAX_ARTIFACT_CONTEXT) {
                  context += `_... (${filesWithContent.length - filesWithContent.indexOf(file)} remaining files omitted due to size limit)_\n\n`;
                  break;
                }

                totalArtifactSize += contentSize;

                // Detect language from file path or metadata
                const ext = file.path.split('.').pop() || '';
                const language = file.metadata?.language || ext || 'text';

                context += `**${file.path}**\n\`\`\`${language}\n${file.content}\n\`\`\`\n\n`;
              }
            }
          } catch (e) {
            console.error('Failed to parse artifact content:', e);
            // If content isn't JSON, treat as plain text
            context += `### ${artifact.name}\n\`\`\`\n${artifact.content}\n\`\`\`\n\n`;
          }
        }
      }

      context += '---\n\n';
    });

    context += 'Help the user understand these results and suggest improvements.';
    return context;
  }, [runData]);

  const handleSendMessage = async () => {
    if (!input.trim() || streaming) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setStreaming(true);

    // Add empty assistant message that will be populated
    const assistantMessageIndex = messages.length + 1;
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch(apiClient.baseUrl + '/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: model,
          api_key: apiKey,
          messages: [...messages, userMessage].map(m => ({ role: m.role, content: m.content })),
          system_message: systemMessage
        })
      });

      if (!response.ok) {
        // Try to get error details from response
        let errorMessage = `HTTP ${response.status}`;
        try {
          const errorData = await response.json();
          if (typeof errorData === 'string') {
            errorMessage = errorData;
          } else if (errorData.detail) {
            // FastAPI validation errors
            if (Array.isArray(errorData.detail)) {
              errorMessage = errorData.detail.map(err => `${err.loc?.join('.')}: ${err.msg}`).join(', ');
            } else {
              errorMessage = errorData.detail;
            }
          } else if (errorData.message) {
            errorMessage = errorData.message;
          } else {
            errorMessage = JSON.stringify(errorData, null, 2);
          }
        } catch {
          // If response isn't JSON, use status text
          errorMessage = `${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const reader = response.body.getReader();
      // eslint-disable-next-line no-undef
      const decoder = new TextDecoder();
      let accumulatedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);

              // LiteLLM returns OpenAI-format responses
              if (parsed.choices?.[0]?.delta?.content) {
                accumulatedContent += parsed.choices[0].delta.content;
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[assistantMessageIndex] = { role: 'assistant', content: accumulatedContent };
                  return newMessages;
                });
              }

              // Handle error from proxy
              if (parsed.error) {
                throw new Error(typeof parsed.error === 'string' ? parsed.error : JSON.stringify(parsed.error));
              }
            } catch (e) {
              if (e.message && !e.message.includes('Unexpected token')) {
                console.error('Streaming error:', e);
                throw e;
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMsg = error.message || 'Unknown error occurred';
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[assistantMessageIndex] = {
          role: 'error',
          content: errorMsg
        };
        return newMessages;
      });
    } finally {
      setStreaming(false);
    }
  };

  const handleSaveSettings = (newModel, newApiKey) => {
    localStorage.setItem('llm_model', newModel);
    localStorage.setItem('llm_api_key', newApiKey);
    setModel(newModel);
    setApiKey(newApiKey);
    setShowSetup(false);
  };

  const handleChangeSettings = () => {
    setShowSetup(true);
  };

  // Empty state
  if (!selectedRunIds || selectedRunIds.length === 0) {
    return null;
  }

  // Loading state
  if (loading) {
    return (
      <div className="chat-tab">
        <div className="chat-tab-loading">
          <p>Loading run data...</p>
        </div>
      </div>
    );
  }

  // Chat interface
  return (
    <>
      {/* Settings Modal */}
      {showSetup && (
        <div className="settings-modal-overlay" onClick={() => setShowSetup(false)}>
          <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
            <div className="settings-modal-header">
              <h2>LLM Configuration</h2>
              <button className="settings-close-btn" onClick={() => setShowSetup(false)}>×</button>
            </div>
            <div className="settings-modal-body">
              <form onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                handleSaveSettings(formData.get('model'), formData.get('apiKey'));
              }}>
                <div className="form-group">
                  <label htmlFor="model">Model</label>
                  <input
                    id="model"
                    name="model"
                    type="text"
                    placeholder="gpt-4o-mini, claude-3-5-sonnet-20241022, ollama/llama2"
                    className="model-input"
                    defaultValue={model}
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="apiKey">API Key</label>
                  <input
                    id="apiKey"
                    name="apiKey"
                    type="password"
                    placeholder="sk-... or sk-ant-..."
                    className="api-key-input"
                    defaultValue={apiKey}
                  />
                </div>

                <div className="setup-actions">
                  <button type="submit" className="btn-primary">
                    {apiKey ? 'Save Changes' : 'Save & Start Chatting'}
                  </button>
                  {apiKey && (
                    <button type="button" className="btn-secondary" onClick={() => setShowSetup(false)}>
                      Cancel
                    </button>
                  )}
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    <div className="chat-tab">
      <div className="chat-messages" ref={messagesContainerRef} onScroll={handleScroll}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`message message-${msg.role}`}>
            <div className="message-content">
              {msg.role === 'error' ? (
                <div className="error-message">
                  <strong>⚠️ Error:</strong> {msg.content}
                </div>
              ) : msg.role === 'assistant' ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                >
                  {msg.content}
                </ReactMarkdown>
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} style={{ paddingBottom: '2rem' }} />
      </div>

      {/* Resize handle */}
      <div
        className="chat-resize-handle"
        onMouseDown={(e) => {
          e.preventDefault();
          setIsResizing(true);
        }}
      />

      <div className="chat-input-container" style={{ height: `${inputHeight}px` }}>
        <textarea
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSendMessage();
            }
          }}
          placeholder="Ask about these experiment results..."
          disabled={streaming}
        />
        <div className="chat-input-actions">
          <button className="settings-icon-btn" onClick={handleChangeSettings} title="Settings">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
              <circle cx="12" cy="12" r="3"></circle>
            </svg>
          </button>
          <button
            className="chat-send-btn"
            onClick={handleSendMessage}
            disabled={!input.trim() || streaming}
          >
            {streaming ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
    </>
  );
};
