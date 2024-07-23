import {useProject} from '../hooks/projects'
import {HeaderTile} from '../widgets/tiles/project_tile'

export default function ProjectPage(props){
    let {dataset} = props
    let project = useProject(dataset)
    if(project === undefined){
        return <div>Loading...</div>
    }
    return <div>
        <HeaderTile project={project}></HeaderTile>
    </div>
}