import './style.css'
import {useParams} from "react-router-dom";
import {useProject} from "../../hooks/projects";
import {useExperiment} from "../../hooks/experiments";

export default function ProjectPage(props) {
    let {experiment, dataset} = useParams()
    let project = useProject(dataset)
    let experiment_data = useExperiment(project, experiment)
    if(project === undefined){
        return <div>Loading...</div>
    }

    return <>
        {experiment_data.name}
    </>
}