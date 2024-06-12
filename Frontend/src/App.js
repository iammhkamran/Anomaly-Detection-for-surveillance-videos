import './App.css';
import React from 'react';
import Sidebar from './Components/Sidebar';
import RecordedVideo from './Components/RecordedVideo';
import LiveClassification from './Components/LiveClassification';
import Logs from './Components/Logs';

import { createBrowserRouter, RouterProvider } from "react-router-dom";

function App() {
  const [videoClass, setVideoClass] = React.useState("")

  const router = createBrowserRouter([
    {
      path: "/",
      element: <><Sidebar /><RecordedVideo videoClass={videoClass} /></>,
    },
    {
      path: "/Live-Classification",
      element: <><Sidebar /><LiveClassification videoClass={videoClass} /></>,
    },
    {
      path: "/Logs",
      element: <><Sidebar /><Logs /></>,
    },
  ]);

  return (
    <div className="App">
      <RouterProvider router={router} />
    </div>
  );
}

export default App;
