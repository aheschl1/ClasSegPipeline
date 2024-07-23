import './App.css';
import ProjectPage from "./pages/project/project";
import Home from "./pages/home/home";
import {Route, Routes} from "react-router-dom";
import About from "./pages/about";
import ExperimentPage from "./pages/experiment/experiment";

function App() {
    return <>
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path={"/projects/:dataset"} element={<ProjectPage/>}/>
            <Route path={"/projects/:dataset/experiments/:experiment"} element={<ExperimentPage/>}/>
        </Routes>
    </>
}

export default App;
