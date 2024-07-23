import logo from './logo.svg';
import './App.css';
import {useProjects} from "./hooks/projects.js";
import ProjectTile from './widgets/tiles/project_tile'
import {useState} from "react";
import ProjectPage from "./project_page/ProjectPage";

function App() {
    let projects = useProjects();
    let [selectedProject, setSelectedProject] = useState(undefined);
    const inspectProject = (project)=>setSelectedProject(project)

    if(selectedProject !== undefined){
        return <ProjectPage dataset={selectedProject}/>
    }
    return (
        <div className="App">
            <div className="ProjectTiles">
                {projects.map((p)=>
                    <ProjectTile onClick={()=>inspectProject(p.name)} project={p} key={p.name}/>
                )}
            </div>
        </div>
    );
}

export default App;
