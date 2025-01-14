import { WorkerCommand, WorkerResponse, AsciiOperationParams, AsciiConfig } from "./const";

const ctx: Worker = self as any; // eslint-disable-line no-restricted-globals
let config: AsciiConfig | null = null;
const contrastFactor = (259 * (128 + 255)) / (255 * (259 - 128));

const predict = async (config: AsciiConfig, params: AsciiOperationParams, data: Uint8ClampedArray) => {
    const asciiStr = params.asciiStr;
    const fontSize = params.fontSize;
    const asciiCharacters = asciiStr.split("");

    const imageData = new ImageData(data, params.processWidth, params.processHeight);
    // ImageData作成
    const offscreen = new OffscreenCanvas(imageData.width, imageData.height);
    const ctx = offscreen.getContext("2d")!;
    ctx.font = fontSize + 'px "Courier New", monospace';
    ctx.textBaseline = "top";
    ctx.putImageData(imageData, 0, 0);
    const m = ctx.measureText(asciiStr);
    const charWidth = Math.floor(m.width / asciiCharacters.length);
    const tmpWidth = Math.ceil(imageData.width / charWidth);
    const tmpHeight = Math.ceil(imageData.height / fontSize);

    // Generate Image for Brightness
    const offscreenForBrightness = new OffscreenCanvas(tmpWidth, tmpHeight);
    const brCtx = offscreenForBrightness.getContext("2d")!;
    brCtx.drawImage(offscreen, 0, 0, tmpWidth, tmpHeight);
    const brImageData = brCtx.getImageData(0, 0, tmpWidth, tmpHeight);

    // generate chars agaist the each dot
    const lines: string[] = [];
    for (let y = 0; y < tmpHeight; y++) {
        let line = "";
        for (let x = 0; x < tmpWidth; x++) {
            const offset = (y * tmpWidth + x) * 4;
            const r = Math.max(0, Math.min(Math.floor((brImageData.data[offset + 0] - 128) * contrastFactor) + 128, 255));
            const g = Math.max(0, Math.min(Math.floor((brImageData.data[offset + 1] - 128) * contrastFactor) + 128, 255));
            const b = Math.max(0, Math.min(Math.floor((brImageData.data[offset + 2] - 128) * contrastFactor) + 128, 255));

            var brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
            var character = asciiCharacters[asciiCharacters.length - 1 - Math.round(brightness * (asciiCharacters.length - 1))];
            line += character;
        }
        lines.push(line);
    }
    return lines;
};

onmessage = async (event) => {
    //  console.log("event", event)
    if (event.data.message === WorkerCommand.INITIALIZE) {
        config = event.data.config as AsciiConfig;
        ctx.postMessage({ message: WorkerResponse.INITIALIZED });
    } else if (event.data.message === WorkerCommand.PREDICT) {
        const params: AsciiOperationParams = event.data.params;
        const data: Uint8ClampedArray = event.data.data;

        const prediction = await predict(config!, params, data);
        ctx.postMessage({
            message: WorkerResponse.PREDICTED,
            prediction: prediction,
        });
    } else {
        console.log("not implemented");
    }
};
