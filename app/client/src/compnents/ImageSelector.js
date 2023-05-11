import React, { useState } from "react";

const ImageAreaSelector = ({ imageLink = "" }) => {
  const [startX, setStartX] = useState(null);
  const [startY, setStartY] = useState(null);
  const [endX, setEndX] = useState(null);
  const [endY, setEndY] = useState(null);
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState(false);

  const handleMouseDown = (event) => {
    const imgRect = event.target.getBoundingClientRect();
    const startX = event.clientX - imgRect.left;
    const startY = event.clientY - imgRect.top;

    setStartX(startX);
    setStartY(startY);
    setEndX(startX);
    setEndY(startY);
    setSelecting(true);
    setSelected(true);
  };

  const handleMouseMove = (event) => {
    if (!selecting) return;

    const imgRect = event.target.getBoundingClientRect();
    const currentX = event.clientX - imgRect.left;
    const currentY = event.clientY - imgRect.top;

    setEndX(currentX);
    setEndY(currentY);
  };

  const handleMouseUp = () => {
    setSelecting(false);
  };

  const getAreaDimensions = () => {
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);
    const width = Math.abs(endX - startX);
    const height = Math.abs(endY - startY);

    return { x, y, width, height };
  };

  const handleClearSelection = () => {
    setSelected(false);
    setStartX(null);
    setStartY(null);
    setEndX(null);
    setEndY(null);
  };

  const handleDataSubmission = () => {};
  return (
    <div className="flex flex-col w-full justify-center items-center space-y-10 ">
      <div>
        <h1 className="text-3xl font-bold">Select your name</h1>
      </div>
      <div className="flex mx-auto space-x-24">
        <div className="w-24">
          <p>X: {getAreaDimensions().x}</p>
          <p>Y: {getAreaDimensions().y}</p>
        </div>
        <div className="w-24">
          <p>Width: {getAreaDimensions().width}</p>
          <p>Height: {getAreaDimensions().height}</p>
        </div>
      </div>

      <div style={{ position: "relative", display: "inline-block" }}>
        {imageLink ? (
          <img
            src={imageLink}
            alt="s"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            style={{ cursor: "crosshair" }}
            onDragStart={(e) => {
              e.preventDefault();
            }}
          />
        ) : (
          <div className="w-96 h-96 bg-slate-200 animate-pulse" />
        )}
        {selected && (
          <div
            className="bg-red-600 bg-opacity-25"
            style={{
              position: "absolute",
              top: getAreaDimensions().y,
              left: getAreaDimensions().x,
              width: getAreaDimensions().width,
              height: getAreaDimensions().height,
              border: "1px dashed white",
              pointerEvents: "none",
            }}
          />
        )}
      </div>

      <div className="flex space-x-10">
        <div className="">
          <button
            className="bg-green-500 hover:bg-green-600 transition-all w-24 py-2 text-white rounded-lg"
            onClick={handleDataSubmission}
          >
            Submit
          </button>
        </div>
        <div className="">
          <button
            className="bg-red-500 hover:bg-red-600 transition-all w-24 py-2 text-white rounded-lg"
            onClick={handleClearSelection}
          >
            Reselect
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImageAreaSelector;
