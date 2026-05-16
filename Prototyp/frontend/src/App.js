import { useEffect, useState } from "react";

function App() {
  const [message, setMessage] = useState("");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/hello")
      .then((response) => response.json())
      .then((data) => setMessage(data.message))
      .catch((error) => console.error("Błąd:", error));
  }, []);

  return (
    <div>
      <h1>Frontend React</h1>
      <p>Odpowiedź z backendu: {message}</p>
    </div>
  );
}

export default App;