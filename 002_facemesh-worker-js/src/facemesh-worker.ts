import { BackendTypes, FacemeshConfig, FacemeshFunctionTypes, FacemeshOperationParams, FaceMeshPredictionEx, FacemeshPredictionMediapipe, ModelTypes, TRIANGULATION } from "./const";
import * as tf from "@tensorflow/tfjs";
export { FacemeshConfig, FacemeshOperationParams, NUM_KEYPOINTS, TRIANGULATION, BackendTypes, ModelTypes, FaceMeshPredictionEx } from "./const";
import * as faceLandmarksDetectionCurrent from "@tensorflow-models/face-landmarks-detection";

// @ts-ignore
import workerJs from "worker-loader?inline=no-fallback!./facemesh-worker-worker.ts";
import { getBrowserType, LocalWorker, WorkerManagerBase } from "@dannadori/worker-base";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";

import * as faceMesh from "@mediapipe/face_mesh";
import { Face, Keypoint } from "@tensorflow-models/face-landmarks-detection/dist/types";
import { BoundingBox } from "@tensorflow-models/face-landmarks-detection/dist/shared/calculators/interfaces/shape_interfaces";
export { BoundingBox } from "@tensorflow-models/face-landmarks-detection/dist/shared/calculators/interfaces/shape_interfaces";
export { Face, Keypoint } from "@tensorflow-models/face-landmarks-detection/dist/types";

export const generateFacemeshDefaultConfig = (): FacemeshConfig => {
    const defaultConf: FacemeshConfig = {
        browserType: getBrowserType(),
        backendType: BackendTypes.WebGL,
        modelReloadInterval: 1024 * 60,
        model: {
            maxContinuousChecks: 5,
            detectionConfidence: 0.9,
            maxFaces: 10,
            iouThreshold: 0.3,
            scoreThreshold: 0.75,
            refineLandmarks: false,
        },
        processOnLocal: true,
        wasmPaths: {
            "tfjs-backend-wasm.wasm": "/tfjs-backend-wasm.wasm",
            "tfjs-backend-wasm-simd.wasm": "/tfjs-backend-wasm-simd.wasm",
            "tfjs-backend-wasm-threaded-simd.wasm": "/tfjs-backend-wasm-threaded-simd.wasm",
        },
        pageUrl: window.location.href,
        modelType: ModelTypes.mediapipe,
    };
    return defaultConf;
};

export const generateDefaultFacemeshParams = () => {
    const defaultParams: FacemeshOperationParams = {
        type: FacemeshFunctionTypes.DetectMesh,
        processWidth: 300,
        processHeight: 300,
        predictIrises: false,
        movingAverageWindow: 10,
        // trackingAreaMarginRatioX: 0.3,
        // trackingAreaMarginRatioTop: 0.8,
        // trackingAreaMarginRatioBottom: 0.2,
    };
    return defaultParams;
};

export class LocalFM extends LocalWorker {
    model2: faceLandmarksDetectionCurrent.FaceLandmarksDetector | null = null;
    load_module = async (config: FacemeshConfig) => {
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
            await tf.setBackend("cpu");
        } else {
            console.log("use webgl backend");
            await tf.setBackend("webgl");
        }
    };

    init = async (config: FacemeshConfig) => {
        await this.load_module(config);
        await tf.ready();

         if (config.modelType === ModelTypes.mediapipe) {
            try {
                this.model2?.dispose();
            } catch (error) {
                console.log("this error is ignored", error)
            }
            this.model2 = await faceLandmarksDetectionCurrent.createDetector(faceLandmarksDetectionCurrent.SupportedModels.MediaPipeFaceMesh, {
                runtime: "mediapipe",
                refineLandmarks: config.model.refineLandmarks,
                maxFaces: config.model.maxFaces,
                solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@${faceMesh.VERSION}`,
            });
        } else {
            try {
                this.model2?.dispose();
            } catch (error) {
                console.log("this error is ignored", error)
            }
            this.model2 = await faceLandmarksDetectionCurrent.createDetector(faceLandmarksDetectionCurrent.SupportedModels.MediaPipeFaceMesh, {
                runtime: "tfjs",
                refineLandmarks: config.model.refineLandmarks,
                maxFaces: config.model.maxFaces,
            });
        }

        console.log("facemesh loaded locally", config);
    };

    predict = async (config: FacemeshConfig, params: FacemeshOperationParams, targetCanvas: HTMLCanvasElement): Promise< Face[] | null> => {
        // console.log("Loacal BACKEND:", tf.getBackend());
        const ctx = targetCanvas.getContext("2d")!;
        const newImg = ctx.getImageData(0, 0, targetCanvas.width, targetCanvas.height);
       if (this.model2) {
            const prediction = await this.model2.estimateFaces(newImg, { flipHorizontal: false });
            return prediction;
        } else {
            return null;
        }
    };
}

export class FacemeshWorkerManager extends WorkerManagerBase {
    private config = generateFacemeshDefaultConfig();
    localWorker = new LocalFM();

    init = async (config: FacemeshConfig | null) => {
        this.config = config || generateFacemeshDefaultConfig();
        await this.initCommon(
            {
                useWorkerForSafari: true,
                processOnLocal: this.config.processOnLocal,
                workerJs: () => {
                    return new workerJs();
                },
            },
            config
        );
        return;
    };

    predict = async (params: FacemeshOperationParams, targetCanvas: HTMLCanvasElement | HTMLVideoElement): Promise<FaceMeshPredictionEx> => {
        const currentParams = { ...params };
        const resizedCanvas = this.generateTargetCanvas(targetCanvas, currentParams.processWidth, currentParams.processHeight);
        if (!this.worker) {
            const prediction = await this.localWorker.predict(this.config, currentParams, resizedCanvas);
            return this.generatePredictionEx(this.config, params, prediction);
        }
        const imageData = resizedCanvas.getContext("2d")!.getImageData(0, 0, resizedCanvas.width, resizedCanvas.height);
        const prediction = (await this.sendToWorker(currentParams, imageData.data)) as Face[];
        return this.generatePredictionEx(this.config, params, prediction);
    };

    facesMV: Face[][] = [];

    generatePredictionEx = (config: FacemeshConfig, params: FacemeshOperationParams, prediction:  Face[] | null): FaceMeshPredictionEx => {
            const faces = prediction as Face[] | null;
            const predictionEx: FacemeshPredictionMediapipe = {
                modelType: config.modelType,
                rowPrediction: faces,
            };
            if (params.movingAverageWindow > 0 && faces && faces.length > 0) {
                /// (1)蓄積データ 更新
                if (faces) {
                    while (this.facesMV.length > params.movingAverageWindow) {
                        this.facesMV.shift();
                    }
                }
                if (faces && faces[0] && faces[0].keypoints) {
                    this.facesMV.push(faces);
                }

                /// (2) キーポイント移動平均算出
                /// (2-1) ウィンドウ内の一人目のランドマークを抽出
                const keypointsEach = this.facesMV.map((pred) => {
                    return pred[0].keypoints;
                });
                /// (2-2) 足し合わせ
                const summedKeypoints = keypointsEach.reduce((prev, cur) => {
                    for (let i = 0; i < cur.length; i++) {
                        if (prev[i]) {
                            prev[i][0] = prev[i][0] + cur[i].x;
                            prev[i][1] = prev[i][1] + cur[i].y;
                            prev[i][2] = prev[i][2] + cur[i].z!;
                        } else {
                            prev.push([cur[i].x, cur[i].y, cur[i].z!]);
                        }
                    }
                    return prev;
                }, [] as any);
                /// (2-3) 平均化
                for (let i = 0; i < summedKeypoints.length; i++) {
                    summedKeypoints[i][0] = summedKeypoints[i][0] / this.facesMV.length;
                    summedKeypoints[i][1] = summedKeypoints[i][1] / this.facesMV.length;
                    summedKeypoints[i][2] = summedKeypoints[i][2] / this.facesMV.length;
                }
                /// (2-4) 追加
                predictionEx.singlePersonKeypointsMovingAverage = summedKeypoints;

                /// (3) ボックス移動平均算出
                /// (3-1) ウィンドウ内の一人目のランドマークを抽出
                const boundingBoxEach = this.facesMV.map((pred) => {
                    return pred[0].box;
                });
                /// (2-2) 足し合わせ
                const summedBoundingBox = boundingBoxEach.reduce((prev, cur) => {
                    if (prev.width) {
                        prev.width = prev.width + cur.width;
                        prev.xMax = prev.xMax + cur.xMax;
                        prev.xMin = prev.xMin + cur.xMin;
                        prev.height = prev.height + cur.height;
                        prev.yMax = prev.yMax + cur.yMax;
                        prev.yMin = prev.yMin + cur.yMin;
                    } else {
                        return {
                            width: cur.width,
                            xMax: cur.xMax,
                            xMin: cur.xMin,
                            height: cur.height,
                            yMax: cur.yMax,
                            yMin: cur.yMin,
                        };
                    }
                    return prev;
                }, {} as BoundingBox);
                /// (2-3) 平均化
                console.log();
                summedBoundingBox.width /= this.facesMV.length;
                summedBoundingBox.xMax /= this.facesMV.length;
                summedBoundingBox.xMin /= this.facesMV.length;
                summedBoundingBox.height /= this.facesMV.length;
                summedBoundingBox.yMax /= this.facesMV.length;
                summedBoundingBox.yMin /= this.facesMV.length;
                /// (2-4) 追加
                predictionEx.singlePersonBoxMovingAverage = summedBoundingBox;

                // /// (4)Tracking Area
                // const trackingAreaCenterX = (summedBoundingBox.xMax + summedBoundingBox.xMin) / 2;
                // const trackingAreaCenterY = (summedBoundingBox.yMax + summedBoundingBox.yMin) / 2;

                // predictionEx.trackingArea = {
                //     centerX: trackingAreaCenterX,
                //     centerY: trackingAreaCenterY,
                //     xMin: summedBoundingBox.xMin,
                //     xMax: summedBoundingBox.xMax,
                //     yMin: summedBoundingBox.yMin,
                //     yMax: summedBoundingBox.yMax,
                // };
            }

            return predictionEx;
    };

}

