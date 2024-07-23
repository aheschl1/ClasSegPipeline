import Navbar from "../components/navbar/navbar";


export default function About(){
    return <>
        <Navbar/>
        <h1>
            ClasSeg
        </h1>
        <h3>
            ClasSeg is a python library for training and deploying deep learning models for image classification, segmentation, and other SSL tasks.
        </h3>
        <a href={"https://github.com/aheschl1/ClasSegPipeline"}>
            GitHub Repository
        </a>
    </>
}