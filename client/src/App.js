import './App.css';
import ProjectPage from "./pages/project/project";
import Home from "./pages/home/home";
import {Route, Routes} from "react-router-dom";
import About from "./pages/about";

function App() {
    return <>
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path={"/projects/:dataset"} element={<ProjectPage/>}/>
        </Routes>
    </>
}

export default App;
