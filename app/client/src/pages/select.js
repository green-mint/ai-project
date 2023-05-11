import React from "react";
import ImageSelector from "../compnents/ImageSelector";
import { useParams } from "react-router-dom";

const Select = () => {
  const { img } = useParams();

  return (
    <div className="flex h-screen w-screen justify-center items-center">
      <ImageSelector imageLink={`http://localhost:5000/images/${img}`} />
    </div>
  );
};

export default Select;
