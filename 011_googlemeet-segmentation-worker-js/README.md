# Google meet person segmentation

![image](https://user-images.githubusercontent.com/48346627/104487132-0b101180-5610-11eb-8182-b1be3470c9c9.png)

## Install

```
## install
$ npm install @dannadori/googlemeet-segmentation-worker-js

```

## step by step usage

examples described here is in [this repository](https://github.com/w-okada/image-analyze-workers-examples).

### create workspace and install modules

```
$ mkdir example
$ cd example/
$ mkdir public
$ mkdir src
$ npm init -y
$ npm install @dannadori/googlemeet-segmentation-worker-js
$ npm install html-webpack-plugin react react-dom
$ npm install -D typescript ts-loader webpack webpack-cli webpack-dev-server @types/react @types/react-dom
```

### setup workspace

Copy files below from [this repository](https://github.com/w-okada/image-analyze-workers-examples/tree/master/011_googlemeet-segmentation-worker-js/example).

-   package.json
-   tsconfig.json
-   webpack.config.js

### prepaire public folder

Create index.html in the public folder. The sample of index.html is here.

```
<!DOCTYPE html>
<html lang="ja" style="width: 100%; height: 100%; overflow: hidden">
  <head>
    <meta charset="utf-8" />
    <title>exampleApp</title>
  </head>
  <body style="width: 100%; height: 100%; margin: 0px">
    <noscript>
      <strong>please enable javascript</strong>
    </noscript>
    <div id="app" style="width: 100%; height: 100%" />
  </body>
</html>
```

Then, put image into the public folder

### Edit src/index.tsx

```
import * as React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

const container = document.getElementById("app")!;
const root = createRoot(container);
root.render(<App />);
```

### Edit src/App.tsx

```
import {
  generateDefaultGoogleMeetSegmentationParams,
  generateGoogleMeetSegmentationDefaultConfig,
  GoogleMeetSegmentationWorkerManager,
} from "@dannadori/googlemeet-segmentation-worker-js";
import React from "react";
import { useEffect } from "react";

const config = (() => {
  const c = generateGoogleMeetSegmentationDefaultConfig();
  c.modelKey = "160x96"; // option: "160x96", "128x128", "256x144", "256x256"
  c.processOnLocal = false; // if you want to run prediction on webworker, set false.
  c.useSimd = true; // if you want to use simd, set true.
  return c;
})();

const params = (() => {
  const p = generateDefaultGoogleMeetSegmentationParams();
  p.processWidth = 512; // processing image width,  should be under 1000.
  p.processHeight = 512; // processing image height, should be under 1000.
  return p;
})();

const App = () => {
  useEffect(() => {
    const input = document.getElementById("input") as HTMLImageElement;
    const output = document.getElementById("output") as HTMLCanvasElement;
    const tmpCanvas = document.createElement("canvas");

    const manager = new GoogleMeetSegmentationWorkerManager();
    manager.init(config).then(() => {
      tmpCanvas.width = input.width;
      tmpCanvas.height = input.height;
      tmpCanvas
        .getContext("2d")!
        .drawImage(input, 0, 0, tmpCanvas.width, tmpCanvas.height);

      console.log(tmpCanvas);
      manager.predict(params, tmpCanvas).then((prediction) => {
        console.log(prediction);
        if (!prediction) {
          return;
        }
        output.width = params.processWidth;
        output.height = params.processHeight;
        const mask = new ImageData(
          prediction,
          params.processWidth,
          params.processHeight
        );
        const outputCtx = output.getContext("2d")!;
        outputCtx.putImageData(mask, 0, 0);
        outputCtx.globalCompositeOperation = "source-atop";
        outputCtx.drawImage(tmpCanvas, 0, 0, output.width, output.height);
      });
    });
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "row" }}>
      <div>
        <img id="input" src="./test.jpg"></img>
      </div>
      <div>
        <canvas id="output"></canvas>
      </div>
    </div>
  );
};

export default App;

```

### build and start

```
$ npm run start
```

## Parameters

### Config

You can configure the module by setting configuration created at `const c = generateGoogleMeetSegmentationDefaultConfig(); ` above smaple.

```
export interface MediapipeMixConfig {
    processOnLocal: boolean;
}

export interface GoogleMeetSegmentationConfig {
    processOnLocal: boolean;  // Run the module on webworker or local thread(main thread).
    modelKey: string; // which model you want to use. "160x96", "128x128", "256x144", "256x256"
    useSimd: boolean; // use wasm-simd or normal wasm.
}
```

### Parameter

You can configure the operation by setting paramters created at `const p = generateDefaultGoogleMeetSegmentationParams();` above smaple.

```
export interface GoogleMeetSegmentationOperationParams {
    jbfD: number;                      // Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
    jbfSigmaC: number;                 // Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
    jbfSigmaS: number;                 // Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace .
    jbfPostProcess: PostProcessTypes;  // PostProcess type(0: none, 1: softmax, 2:joint bilateral filter, 3: both softmax and joint bilateral filter)

    threshold: number; // Threshold
    interpolation: InterpolationTypes; // interpolation when enlarge the mask from model output to input process size.

    processWidth: number; // image width pose prediction. up to 1024
    processHeight: number;// image height prediction. up to 1024
}
```
