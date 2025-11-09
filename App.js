import React, {useState} from 'react';
import axios from 'axios';

function App(){
  const [file, setFile] = useState(null);
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);

  const upload = async () => {
    if(!file) return alert("Select an image file");
    setLoading(true);
    const f = new FormData();
    f.append("file", file);
    try {
      const r = await axios.post("http://localhost:5000/predict", f, {headers: {"Content-Type":"multipart/form-data"}});
      setRes(r.data);
    } catch(err){
      alert("Error: " + (err.response?.data?.error || err.message));
    } finally { setLoading(false); }
  }

  return (
    <div style={{padding:20, fontFamily:'Arial'}}>
      <h2>SmartVision - Image Classifier</h2>
      <input type="file" accept="image/*" onChange={e=>setFile(e.target.files[0])}/>
      <button onClick={upload} style={{marginLeft:10}}>Predict</button>
      {loading && <p>Processing...</p>}
      {res && (
        <div style={{marginTop:15}}>
          <b>Class:</b> {res.class} <br/>
          <b>Confidence:</b> {(res.confidence*100).toFixed(2)}%
        </div>
      )}
    </div>
  )
}
export default App;
