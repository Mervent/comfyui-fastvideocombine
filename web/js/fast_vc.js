import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { applyTextReplacements } from "../../../scripts/utils.js";

const NODE_NAME = "FastVideoCombine";

function chainCallback(object, property, callback) {
    if (object == undefined) return;
    if (property in object && object[property]) {
        const orig = object[property];
        object[property] = function () {
            const r = orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r;
        };
    } else {
        object[property] = callback;
    }
}

function fitHeight(node) {
    node.setSize([
        node.size[0],
        node.computeSize([node.size[0], node.size[1]])[1],
    ]);
    node?.graph?.setDirtyCanvas(true);
}

function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;
        const container = document.createElement("div");
        const videoEl = document.createElement("video");

        videoEl.controls = true;
        videoEl.loop = true;
        videoEl.muted = true;
        videoEl.style.width = "100%";
        videoEl.hidden = true;

        container.appendChild(videoEl);

        const previewWidget = this.addDOMWidget(
            "videopreview",
            "preview",
            container,
            {
                serialize: false,
                hideOnZoom: false,
                getValue() {
                    return container.value;
                },
                setValue(v) {
                    container.value = v;
                },
            }
        );

        previewWidget.aspectRatio = null;
        previewWidget.computeSize = function (width) {
            if (this.aspectRatio && !container.hidden) {
                let height = (node.size[0] - 20) / this.aspectRatio + 10;
                if (!(height > 0)) height = 0;
                return [width, height];
            }
            return [width, -4];
        };

        videoEl.addEventListener("loadedmetadata", () => {
            if (videoEl.videoWidth && videoEl.videoHeight) {
                previewWidget.aspectRatio =
                    videoEl.videoWidth / videoEl.videoHeight;
                fitHeight(node);
            }
        });

        this.updateParameters = (params) => {
            if (!params) return;
            let urlParams = { ...params, timestamp: Date.now() };
            videoEl.src = api.apiURL(
                "/view?" + new URLSearchParams(urlParams)
            );
            videoEl.hidden = false;
            container.hidden = false;
        };
    });
}

function addDateFormatting(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const widget = this.widgets.find((w) => w.name === "filename_prefix");
        if (!widget) return;
        widget.serializeValue = () => applyTextReplacements(app, widget.value);
    });
}

app.registerExtension({
    name: "FastVideoCombine",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData?.name !== NODE_NAME) return;

        addDateFormatting(nodeType);
        addVideoPreview(nodeType);

        chainCallback(nodeType.prototype, "onExecuted", function (message) {
            if (message?.gifs) {
                this.updateParameters(message.gifs[0]);
            }
        });
    },
});
