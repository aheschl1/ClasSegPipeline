import logo from './logo.svg';
import './App.css';
import {useProjects} from "./hooks/projects.js";
import ProjectTile from './widgets/tiles/project_tile'

function inspectProject(projectName){
    console.log(projectName)
}

function App() {
    let projects = useProjects();
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
