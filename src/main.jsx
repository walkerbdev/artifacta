import React from 'react';
import ReactDOM from 'react-dom/client';
import Workspace from '@/app/pages/Workspace';
import '@/utils/debugLogger'; // Initialize debug logger if enabled
import { initConsoleFilter } from '@/utils/consoleFilter';
import '@/app/styles/sidebarConstants.css'; // Sidebar component styling constants

// Filter third-party console warnings
if (import.meta.env.VITE_DEBUG_LOGS !== 'true') {
  initConsoleFilter();
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <Workspace />
);
