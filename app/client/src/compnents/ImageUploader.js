import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const ImageUploader = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [image, setImage] = useState(null);
  const navigate = useNavigate();

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImage(file);
    setSelectedImage(URL.createObjectURL(file));
  };

  const handleSendImage = () => {
    const formData = new FormData();
    formData.append("image", image);

    fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        return response.text();
      })
      .then((result) => {
        console.log(result);
        return navigate(`/select/${result}`);
      });
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-10">
      <h1 className="text-3xl font-bold">Upload your CNIC</h1>

      <div className="flex justify-center">
        <input
          type="file"
          id="imageUpload"
          accept="image/*"
          onChange={handleImageUpload}
        />
      </div>
      <div className="flex justify-center items-center w-full max-w-3xl h-96 border">
        {selectedImage ? (
          <img
            className="object-contain max-w-full max-h-full"
            src={selectedImage}
            alt="Selected"
          />
        ) : (
          <div>Your Image here</div>
        )}
      </div>
      <div className="">
        <button
          className="bg-green-500 hover:bg-green-600 transition-all px-4 py-2 text-white rounded-lg"
          onClick={handleSendImage}>
          Save
        </button>
      </div>
    </div>
  );
};

export default ImageUploader;
