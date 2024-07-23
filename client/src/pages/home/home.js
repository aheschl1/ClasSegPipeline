import {useProjects} from "../../hooks/projects.js";
import ProjectTile from '../../components/project_tile/project_tile'
import {useNavigate} from "react-router-dom";
import './style.css';
import Navbar from "../../components/navbar/navbar";

export default function Home() {
    let navigate = useNavigate();
    let projects = useProjects();

    return (
        <>
            <Navbar/>
            <div className="ProjectTiles">
                {projects.map((p)=>
                    <ProjectTile onClick={()=>navigate(`/projects/${p.name}`)} project={p} key={p.name}/>
                )}
            </div>
        </>
    );
}