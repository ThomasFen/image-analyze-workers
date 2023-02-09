//import * as facemesh from '@tensorflow-models/facemesh'
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";

import { BackendTypes, FacemeshConfig, FacemeshOperationParams, ModelTypes, WorkerCommand, WorkerResponse } from "./const";
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import { Face } from '@tensorflow-models/face-landmarks-detection/dist/types';
import * as faceMesh from "@mediapipe/face_mesh";
const ctx: Worker = self as any; // eslint-disable-line no-restricted-globals

let model2: faceLandmarksDetection.FaceLandmarksDetector | null = null;

let config: FacemeshConfig | null = null;

const load_module = async (config: FacemeshConfig) => {
    if (config.backendType === BackendTypes.wasm) {
        const dirname = config.pageUrl.substr(0, config.pageUrl.lastIndexOf("/"));
        const wasmPaths: { [key: string]: string } = {};
        Object.keys(config.wasmPaths).forEach((key) => {
            wasmPaths[key] = `${dirname}${config.wasmPaths[key]}`;
        });
        setWasmPaths(wasmPaths);
        console.log("use wasm backend", wasmPaths);
        await tf.setBackend("wasm");
    } else if (config.backendType === BackendTypes.cpu) {
        console.log("use cpu backend");
        await tf.setBackend("cpu");
    } else {
        console.log("use webgl backend");
        await tf.setBackend("webgl");
    }
};

const predict = async (config: FacemeshConfig, params: FacemeshOperationParams, data: Uint8ClampedArray): Promise<Face[]> => {
    const newImg = new ImageData(data, params.processWidth, params.processHeight);

 if (model2) {
        const prediction = await model2.estimateFaces(newImg, { flipHorizontal: false });
        return prediction;
    } else {
        console.log("model not initialized!");
        return [];
    }
};

onmessage = async (event) => {
    if (event.data.message === WorkerCommand.INITIALIZE) {
        console.log("Initialize model!.", event);
        config = event.data.config as FacemeshConfig;
        await load_module(config);
        await tf.ready();
        tf.env().set("WEBGL_CPU_FORWARD", false);

    if (config.modelType === ModelTypes.mediapipe) {
            const prevModel2 = model2;
            try {
                model2 = await faceLandmarksDetection.createDetector(faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, {
                    runtime: "mediapipe",
                    refineLandmarks: config.model.refineLandmarks,
                    maxFaces: config.model.maxFaces,
                    solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@${faceMesh.VERSION}`,
                });
            } catch (error) {
                console.log("error", error);
            }
            ctx.postMessage({ message: WorkerResponse.INITIALIZED });
            try {
                prevModel2?.dispose();
            } catch (error) {
                console.log("this error is ignored", error)
            }
        } else {
            const prevModel2 = model2;
            model2 = await faceLandmarksDetection.createDetector(faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, {
                runtime: "tfjs",
                refineLandmarks: config.model.refineLandmarks,
                maxFaces: config.model.maxFaces,
            });
            ctx.postMessage({ message: WorkerResponse.INITIALIZED });
            prevModel2?.dispose();
        }
    } else if (event.data.message === WorkerCommand.PREDICT) {
        const params = event.data.params as FacemeshOperationParams;
        const data: Uint8ClampedArray = event.data.data;

        const prediction = await predict(config!, params, data);
        ctx.postMessage({
            message: WorkerResponse.PREDICTED,
            prediction: prediction,
        });
    }
};
