import { useState } from "react";
import "./App.css";
import "milligram"

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [page, setPage] = useState("home");
  const [loadingData, setLoadingData] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [tablePreview, setTablePreview] = useState([]);
  const [narrative, setNarrative] = useState("");
  const [context, setContext] = useState("");
  const [message, setMessage] = useState("");

  async function loadAndEmbed() {
    setLoadingData(true);
    setMessage("");
    setTablePreview([]);
  
    try {
      const response = await fetch(`${API_URL}/api/demographics/load-embed`, {
        method: "POST"
      });
    
      const data = await response.json();

      setTablePreview(data.table_preview || []);
      setMessage(data.message || "Data loaded.");
    } catch (error) {
      setMessage("Error while loading data.");
    } finally {
      setLoadingData(false);
    }
  }

  async function generateNarrative() {
    setGenerating(true);
    setMessage("");

    try {
      const response = await fetch(`${API_URL}/api/demographics/generate-narrative`, {
        method: "POST"
      });
    
      const data = await response.json();

      if (data.error) {
        setMessage(data.error);
      } else {
        setNarrative(data.narrative || "");
        setContext(data.context || "");
      }
    } catch (error) {
      setMessage("Error while generating narrative.");
    } finally {
      setGenerating(false);
    }
  }

  if (page === "demographics") {
    return (
      <main className="container">
        <button className="button-clear" onClick={() => setPage("home")}>
          Return
        </button>

        <h1>CSR MVP</h1>
        <p>Chapter: Dempgraphics</p>
          
        <section>
          <h2>Source Data</h2>
          
          <button onClick={loadAndEmbed} disabled={loadingData}>
            {loadingData ? "Loading" : "Load data and create embeddings"}
          </button>

          {message && <p className="message">{message}</p>}

          <div className="box">
            {tablePreview.length === 0 ? (
              <p>No data available</p>
            ): (
              <table>
                <thead>
                  <tr>
                    {Object.keys(tablePreview[0]).map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                  
                <tbody>
                  {tablePreview.map((row, index) => (
                    <tr key={index}>
                      {Object.values(row).map((value, valueIndex) => (
                        <td key={valueIndex}>{String(value)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>


        <section>
          <h2>CSR Narrative</h2>
          
          <button onClick={generateNarrative} disabled={generating}>
            {generating ? "Generating" : "Generate narrative"}
          </button>

          <div className="box narrative">
              {narrative || "Narrative will be available here."}
          </div>

          <div className="box narrative">
              {context || "Context will be available here."}
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="container">
      <header className="center">
        <h1>CSR Narrative Assistant</h1>
        <p>Application supporting creation of a Clinical Study Report Narrative.</p>
      </header>

      <div className="chapters">
        <button className="button" onClick={() => setPage("demographics")}>
          <strong>Demographics</strong>
        </button>

        <button className="button" disabled>
          <strong>Safety</strong>
        </button>

        <button className="button" disabled>
          <strong>Efficacy</strong>
        </button>

        <button className="button" disabled>
          <strong>Disposotion</strong>
        </button>
      </div>
    </main>
  );
}
    
export default App;