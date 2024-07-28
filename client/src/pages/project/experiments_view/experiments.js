import './style.css'
import {useExperiments} from "../../../hooks/experiments";
import {useNavigate} from "react-router-dom";

export default function Experiments(props) {
    let {project} = props
    let experiments = useExperiments(project)
    let navigate = useNavigate()

    return (
        <div>
            {props.children}
            {experiments.map((experiment, i) => {
                return <div key={i} className="experiment">
                    <p>{`Name: ${experiment.name}`}</p>
                    <p>{`Fold: ${experiment.fold}`}</p>
                    <p>{`Checkpoints: ${experiment.checkpoints.length}`}</p>
                    <button onClick={()=>navigate(`/projects/${project.name}/experiments/${experiment.name}`)}>Inspect</button>
                </div>
            })}
        </div>
    )
}