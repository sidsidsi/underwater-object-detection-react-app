// Import dependencies
import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
//import Video from "react-video";
import "./App.css";
import videoSource from "./underwater.mp4"; // Import video source file
import {drawRect} from "./utilities"; // Import drawing utility here
import { loadGraphModel} from "@tensorflow/tfjs";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const detect = async (net) => {
    // Check if video is loaded
    const videoElement = document.getElementById("video");
      if(videoElement.readyState >= 3){
        // Set Video Properties
        const videoWidth = 640;
        const videoHeight = 480;

        // Make Detections
        const detections = tf.tidy (() => {
        const img = tf.browser.fromPixels(document.getElementById("video"))
        const resized = tf.image.resizeBilinear(img, [640,480])
        const casted = resized.cast('int32')
        const expanded = casted.expandDims(0)
        return expanded;
        });
        const obj = await net.executeAsync(detections)

        /* Debugging Detection 
        [0] - Box values [x,y, height, width]
        [2] - Non-post Process Box Classes
        [3] - Actual Box/Label Classes
        [5] - Count of Objects Detected
        [6] - Scores
        */
        console.log(await obj[6].array())
              
        const boxes = await obj[4].array()
        const classes = await obj[2].array()
        const scores = await obj[5].array()
            
        // Draw mesh
        const ctx = canvasRef.current.getContext("2d");
        
        // Update drawing utility for instance, drawSomething(obj, ctx)  
        requestAnimationFrame(()=>{drawRect(boxes[0], classes[0], scores[0], 0.6, videoWidth, videoHeight, ctx)}); 
        
        //reset drawing
        //var canvas = document.getElementById('myCanvas');
        ctx.clearRect(0, 0, 854, 480);

        tf.dispose(detections)
        tf.dispose(obj)
        tf.dispose(net)
        console.log (tf.memory())
      }
  }

  useEffect(()=>{
    // Main function
    const runModel = async () => {
      // Load model network 
      const net = await loadGraphModel('http://192.168.18.61:8080/model.json');
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
    <div class="row no-gutters">
        <div class="col-md-4 no-gutters">
            <div class="leftside">
              <div id="logo"> 
              <img src="https://i.postimg.cc/zBv8ftv8/cawilai-logo.png" alt="" /> 
              </div>
              <section class="header">
              <h1>UNDERWATER</h1>
              </section>
              <section class="subtitle">
              <h2>Object Detection</h2>
              </section>
              <section class="description">
              <h5>300+ images were trained for the <i><b>SSD MobileNet v2 320x320</b></i> model using the <b>TensorFlow Object Detection API</b>.</h5>
              <br></br>
              <br></br>
              <br></br>
              <h5>The expected results are the following:</h5>
              </section>

              <div id="sqr-green">
              <h4 id="result-one">marine-growth</h4>
              </div>
              <div id="sqr-red">
              <h4 id="result-two">anomaly</h4>
              </div>

            </div>
        </div>

        <div class="col-md-8 no-gutters">
            <div class="rightside">
              <div className="videoPlayer">
                <video
                  id="video"
                  src={videoSource}
                  type="video/mp4"
                  autoPlay
                  ref={videoRef}
                  muted={true}
                  style={{
                    borderRadius: 20,
                    position: "absolute",
                    marginLeft: "auto",
                    marginRight: "auto",
                    top:170,
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
                    borderRadius: 20,
                    position: "absolute",
                    marginLeft: "auto",
                    marginRight: "auto",
                    top:170,
                    left: 0,
                    right: 0,
                    textAlign: "center",
                    zindex: 8,
                    width: 854,
                    height: 480,
                  }}
                />
              </div>
              <a href="https://drive.google.com/drive/folders/1jtTjwg24HMTaH15qxpBXCKS_kD1SCL2s?usp=sharing"><span role="img" aria-labelledby="folder">📂</span>Dataset</a>

            </div>
          </div>
    </div>
  );
}

export default App;
