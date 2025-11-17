import React, { useState, useEffect } from "react";
import {
  Upload,
  Leaf,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Brain,
  Zap,
  Sparkles,
  Cpu,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";

// Configurar URL del backend según el entorno
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const DISEASE_INFO = {
  "Black Rot": {
    description: "Enfermedad fúngica que causa lesiones oscuras en hojas.",
    severity: "diseased",
    icon: ShieldAlert,
  },
  ESCA: {
    description:
      "Enfermedad compleja que causa patrones de rayas de tigre y pudrición.",
    severity: "diseased",
    icon: ShieldAlert,
  },
  Healthy: {
    description: "No se detectó enfermedad. La planta parece saludable.",
    severity: "healthy",
    icon: ShieldCheck,
  },
  "Leaf Blight": {
    description:
      "Infección fúngica que causa manchas marrones y deterioro de las hojas.",
    severity: "diseased",
    icon: ShieldAlert,
  },
};

const MODEL_ICONS = {
  convnext: Brain,
  efficientnet: Zap,
  resnet: Cpu,
  vit: Sparkles,
  default: Brain,
};

const getModelIcon = (modelId) => {
  const id = modelId.toLowerCase();
  for (const key in MODEL_ICONS) {
    if (id.includes(key)) {
      return MODEL_ICONS[key];
    }
  }
  return MODEL_ICONS.default;
};

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [currentFile, setCurrentFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [topPrediction, setTopPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await fetch(`${API_URL}/models`);
        const data = await res.json();
        const list = data.models || [];
        setModels(list);
        if (data.default && !selectedModel) {
          setSelectedModel(data.default);
        } else if (list.length > 0 && !selectedModel) {
          setSelectedModel(list[0].id);
        }
      } catch (err) {
        console.warn("Could not load models:", err);
      }
    };
    loadModels();
  }, []);

  useEffect(() => {
    if (currentFile && selectedModel && !isLoading) {
      handleImageSelect(currentFile);
    }
  }, [selectedModel]);

  const handleImageSelect = async (file) => {
    setCurrentFile(file);

    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target.result);
    };
    reader.readAsDataURL(file);

    setIsLoading(true);
    setPredictions(null);
    setTopPrediction(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("top_k", "5");

      const modelQuery = selectedModel
        ? `?model_id=${encodeURIComponent(selectedModel)}`
        : "";
      const response = await fetch(`${API_URL}/predict${modelQuery}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      const preds = data.predictions || data;

      setPredictions(preds);
      if (preds && preds.length > 0) {
        setTopPrediction({
          class: preds[0].label,
          confidence: preds[0].score,
        });
      }
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Error al analizar la imagen. Por favor intenta de nuevo.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleImageSelect(file);
    }
  };

  const handleFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleImageSelect(file);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case "diseased":
        return "text-error";
      case "healthy":
        return "text-success";
      default:
        return "text-muted";
    }
  };

  const getSeverityLabel = (severity) => {
    switch (severity) {
      case "diseased":
        return "Enferma";
      case "healthy":
        return "Sana";
      default:
        return "Desconocido";
    }
  };

  const currentModel = models.find((m) => m.id === selectedModel) || null;

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary">
              <Leaf className="h-7 w-7 text-primary-fg" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                Clasificador de Enfermedades de la Uva
              </h1>
              <p className="text-sm text-muted">
                Detección de enfermedades de plantas con IA
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {models && models.length > 0 && (
          <div className="card mb-8">
            <div className="card-content p-6">
              <h3 className="mb-4 text-sm font-medium text-foreground">
                Seleccionar Modelo de IA
              </h3>
              <div className="model-selector">
                {models.map((model) => {
                  const Icon = getModelIcon(model.id);
                  const isSelected = selectedModel === model.id;
                  return (
                    <button
                      key={model.id}
                      onClick={() => setSelectedModel(model.id)}
                      className={`model-card ${isSelected ? "selected" : ""}`}
                    >
                      <div className="model-icon-wrapper">
                        <Icon className="model-icon" />
                      </div>
                      <div className="model-info">
                        <p className="model-name">{model.name || model.id}</p>
                        {model.labels && (
                          <p className="model-detail">
                            {model.labels.length} clases
                          </p>
                        )}
                      </div>
                      <div
                        className={`model-check ${isSelected ? "visible" : ""}`}
                      >
                        <CheckCircle2 className="h-5 w-5" />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Left Column - Upload Section */}
          <div className="space-y-6">
            <div className="card border-2 border-dashed">
              <div className="card-content p-8">
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  className={`flex flex-col items-center justify-center space-y-4 rounded-lg border-2 border-dashed p-12 transition-colors ${
                    isDragging
                      ? "border-primary bg-primary-light"
                      : "border-border bg-muted-light"
                  }`}
                >
                  <div className="rounded-full bg-primary-light p-4">
                    <Upload className="h-8 w-8 text-primary" />
                  </div>
                  <div className="text-center">
                    <p className="mb-2 text-sm font-medium text-foreground">
                      Arrastra tu imagen aquí o haz clic para buscar
                    </p>
                    <p className="text-xs text-muted">
                      Soporta JPG, PNG (máx 10MB)
                    </p>
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileInput}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="btn btn-primary cursor-pointer"
                  >
                    Seleccionar Imagen
                  </label>
                </div>
              </div>
            </div>

            {selectedImage && (
              <div className="card">
                <div className="card-content p-6">
                  <h3 className="mb-4 text-sm font-medium text-foreground">
                    Imagen Seleccionada
                  </h3>
                  <div className="relative overflow-hidden rounded-lg bg-muted-light">
                    <img
                      src={selectedImage || "/placeholder.svg"}
                      alt="Selected plant leaf"
                      className="h-auto w-full object-contain"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Results Section */}
          <div className="space-y-6">
            {isLoading && (
              <div className="card">
                <div className="card-content p-8">
                  <div className="flex flex-col items-center justify-center space-y-4">
                    <div className="spinner"></div>
                    <p className="text-sm text-muted">Analizando imagen...</p>
                  </div>
                </div>
              </div>
            )}

            {topPrediction && !isLoading && (
              <>
                <div className="card border-2">
                  <div className="card-content p-6">
                    <div className="mb-4 flex items-start justify-between">
                      <div>
                        <p className="text-sm text-muted">Diagnóstico</p>
                        <h2 className="mt-1 text-3xl font-bold text-foreground">
                          {topPrediction.class}
                        </h2>
                      </div>
                      {(() => {
                        const info =
                          DISEASE_INFO[topPrediction.class] ||
                          DISEASE_INFO["Healthy"];
                        const Icon = info.icon;
                        return (
                          <div className="flex flex-col items-center gap-1">
                            <Icon
                              className={`h-12 w-12 ${getSeverityColor(
                                info.severity
                              )}`}
                            />
                            <span
                              className={`text-xs font-medium ${getSeverityColor(
                                info.severity
                              )}`}
                            >
                              {getSeverityLabel(info.severity)}
                            </span>
                          </div>
                        );
                      })()}
                    </div>

                    <div className="space-y-4">
                      <div>
                        <div className="mb-2 flex items-center justify-between text-sm">
                          <span className="text-muted">Confianza</span>
                          <span className="font-semibold text-foreground">
                            {(topPrediction.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{
                              width: `${topPrediction.confidence * 100}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div className="alert">
                        <p className="text-sm leading-relaxed">
                          {
                            (
                              DISEASE_INFO[topPrediction.class] ||
                              DISEASE_INFO["Healthy"]
                            ).description
                          }
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {predictions && predictions.length > 0 && (
                  <div className="card">
                    <div className="card-content p-6">
                      <h3 className="mb-4 text-sm font-medium text-foreground">
                        Todas las Predicciones
                      </h3>
                      <div className="space-y-3">
                        {predictions.map((pred, idx) => (
                          <div key={idx} className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                              <span className="font-medium text-foreground">
                                {pred.label}
                              </span>
                              <span className="text-muted">
                                {(pred.score * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="progress-bar secondary">
                              <div
                                className="progress-fill secondary"
                                style={{ width: `${pred.score * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            {!topPrediction && !isLoading && (
              <div className="card border-dashed">
                <div className="card-content p-12">
                  <div className="flex flex-col items-center justify-center text-center">
                    <div className="mb-4 rounded-full bg-muted-light p-4">
                      <Leaf className="h-8 w-8 text-muted" />
                    </div>
                    <p className="text-sm text-muted">
                      Sube una imagen para comenzar
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="card mt-8">
          <div className="card-content p-6">
            <h3 className="mb-4 text-lg font-semibold text-foreground">
              Acerca de este Clasificador
            </h3>
            <div className="grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-1">
                <p className="font-medium text-foreground">Modelo</p>
                <p className="text-muted">
                  {currentModel ? currentModel.name : "ConvNeXt Tiny"}
                </p>
              </div>
              <div className="space-y-1">
                <p className="font-medium text-foreground">Clases</p>
                <p className="text-muted">
                  {currentModel && currentModel.labels
                    ? `${currentModel.labels.length} tipos de enfermedades`
                    : "4 tipos de enfermedades"}
                </p>
              </div>
              <div className="space-y-1">
                <p className="font-medium text-foreground">Tamaño de Entrada</p>
                <p className="text-muted">224x224 píxeles</p>
              </div>
              <div className="space-y-1">
                <p className="font-medium text-foreground">Entrenado En</p>
                <p className="text-muted">Dataset de hojas de uva</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
