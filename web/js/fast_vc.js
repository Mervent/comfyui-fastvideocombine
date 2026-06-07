import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { applyTextReplacements } from "../../../scripts/utils.js";

const NODE_NAME = "VHS_VideoCombineFast";

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

function addFormatWidgets(nodeType, nodeData) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        let formatWidget = null;
        let formatWidgetIndex = -1;
        for (let i = 0; i < this.widgets.length; i++) {
            if (this.widgets[i].name === "format") {
                formatWidget = this.widgets[i];
                formatWidgetIndex = i + 1;
                break;
            }
        }
        if (!formatWidget) return;

        let formatWidgetsCount = 0;
        chainCallback(formatWidget, "callback", (value) => {
            const formats =
                LiteGraph.registered_node_types[this.type]?.nodeData?.input
                    ?.required?.format?.[1]?.formats;

            let newWidgets = [];
            if (formats?.[value]) {
                for (let wDef of formats[value]) {
                    let type = wDef[2]?.widgetType ?? wDef[1];
                    if (Array.isArray(type)) type = "COMBO";
                    app.widgets[type](this, wDef[0], wDef.slice(1), app);
                    let w = this.widgets.pop();
                    w.config = wDef.slice(1);
                    newWidgets.push(w);
                }
            }

            let removed = this.widgets.splice(
                formatWidgetIndex,
                formatWidgetsCount,
                ...newWidgets
            );
            let newNames = new Set(newWidgets.map((w) => w.name));
            for (let w of removed) {
                w?.onRemove?.();
                if (newNames.has(w.name)) continue;
                let slot = this.inputs.findIndex((i) => i.name == w.name);
                if (slot >= 0) this.removeInput(slot);
            }
            for (let w of newWidgets) {
                let existing = this.inputs.find((i) => i.name == w.name);
                if (!existing) {
                    this.addInput(w.name, w.config[0], {
                        widget: { name: w.name },
                    });
            }
            fitHeight(this);
            formatWidgetsCount = newWidgets.length;
        });
    });
}

function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;
        const container = document.createElement("div");
        const videoEl = document.createElement("video");
        const imgEl = document.createElement("img");

        videoEl.controls = true;
        videoEl.loop = true;
        videoEl.muted = true;
        videoEl.style.width = "100%";
        videoEl.hidden = true;

        imgEl.style.width = "100%";
        imgEl.hidden = true;

        container.appendChild(videoEl);
        container.appendChild(imgEl);

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

        imgEl.addEventListener("load", () => {
            if (imgEl.naturalWidth && imgEl.naturalHeight) {
                previewWidget.aspectRatio =
                    imgEl.naturalWidth / imgEl.naturalHeight;
                fitHeight(node);
            }
        });

        this.updateParameters = (params, force_update) => {
            if (!params) return;
            let urlParams = { ...params, timestamp: Date.now() };

            if (
                params.format?.split("/")[0] === "video" ||
                params.format === "folder"
            ) {
                videoEl.src = api.apiURL(
                    "/view?" + new URLSearchParams(urlParams)
                );
                videoEl.hidden = false;
                imgEl.hidden = true;
                container.hidden = false;
            } else if (params.format?.split("/")[0] === "image") {
                imgEl.src = api.apiURL(
                    "/view?" + new URLSearchParams(urlParams)
                );
                videoEl.hidden = true;
                imgEl.hidden = false;
                container.hidden = false;
            }
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
        addFormatWidgets(nodeType, nodeData);
        addVideoPreview(nodeType);

        chainCallback(nodeType.prototype, "onExecuted", function (message) {
            if (message?.gifs) {
                this.updateParameters(message.gifs[0], true);
            }
        });
    },
});
