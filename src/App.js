// Import dependencies
import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
//import Video from "react-video";
import "./App.css";
import videoSource from "./ShipHull.mp4"; // Import video source file
import {drawRect} from "./utilities"; // Import drawing utility here
import { loadGraphModel } from "@tensorflow/tfjs";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const detect = async (net) => {
    // Check if video is loaded
    const videoElement = document.getElementById("video");
      if(videoElement.readyState >= 3){
        // Get Video Properties
        //const video = videoRef.current.video;
        const videoWidth = 640;
        const videoHeight = 480;
          // const videoWidth = videoRef.current.video.videoWidth;
          // const videoHeight = videoRef.current.video.videoHeight;
        
          // Set video width
          // videoRef.current.video.width = videoWidth;
          // videoRef.current.video.height = videoHeight;
        
        // Set canvas height and width
          // canvasRef.current.width = videoWidth;
          // canvasRef.current.height = videoHeight;

        // Make Detections
        const img = tf.browser.fromPixels(document.getElementById("video"))
        const resized = tf.image.resizeBilinear(img, [640,480])
        const casted = resized.cast('int32')
        const expanded = casted.expandDims(0)
        const obj = await net.executeAsync(expanded)

        /* Debugging Detection 
        [0] - Box valuesd [x,y, height, width]
        [2] - Non-post Process Box Classes
        [3] - Actual Box/Label Classes
        [5] - Count of Objects Detected
        [6] - Scores
        */
        console.log(await obj[6].array())
              
        const boxes = await obj[0].array()
        const classes = await obj[3].array()
        const scores = await obj[6].array()
            
        // Draw mesh
        const ctx = canvasRef.current.getContext("2d");
        
        // Update drawing utility for instance, drawSomething(obj, ctx)  
        requestAnimationFrame(()=>{drawRect(boxes[0], classes[0], scores[0], 0.6, videoWidth, videoHeight, ctx)}); 
        
        tf.dispose(img)
        tf.dispose(resized)
        tf.dispose(casted)
        tf.dispose(expanded)
        tf.dispose(obj)

      }
  }

  useEffect(()=>{
    // Main function
    const runModel = async () => {
      // Load model network 
      const net = await loadGraphModel('http://192.168.1.2:8080/model.json');
      //const net = await tf.loadGraphModel('https://github.com/sidsidsi/underwater-object-detection-initial/blob/main/content/inference_graph/web_model/model.json')
      console.log('model loaded');

      // Loop for detection
      setInterval(() => {
        detect(net);
      }, 16.7);
    };
    runModel();
  }, []);

  return (
    <div className="App">
      <section class="title">
        <h1>🤿 UNDERWATER OBJECT DETECTION</h1>
      </section>  
      <p>A React web app that identifies and labels the <b>marine-growth</b> and <b>anomaly</b> present in the video using the custome trained <b>SSD MobileNet v2 320x320</b>.
      </p>
      <header className="App-header">
        <video
          id="video"
          src={videoSource}
          type="video/mp4"
          autoPlay
          ref={videoRef}
          muted={true}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 854,
            height: 480,
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 8,
            width: 854,
            height: 480,
          }}
        />
      </header>
      <footer>
        <p>John Cedric P. Miguel ©️ CAWIL.ai</p>
      </footer>
    </div>
  );
}

export default App;
