import React, { createContext, useState, useContext } from 'react';

const AppContext = createContext();

export function AppProvider({ children }) {
  const [file, setFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [taskType, setTaskType] = useState('');
  const [previewData, setPreviewData] = useState(null);
  const [automlResults, setAutomlResults] = useState(null);
  const [history, setHistory] = useState([]);

  return (
    <AppContext.Provider value={{
      file, setFile,
      targetColumn, setTargetColumn,
      taskType, setTaskType,
      previewData, setPreviewData,
      automlResults, setAutomlResults,
      history, setHistory
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  return useContext(AppContext);
}
