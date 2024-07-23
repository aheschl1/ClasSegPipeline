import {useProject} from '../../hooks/projects'
import {useParams} from 'react-router-dom'
import './style.css'
import Navbar from "../../components/navbar/navbar";
import ImageView from "./image_views/image_view";
import Experiments from "./experiments_view/experiments";
import {useState} from "react";


const ContentBlock = (props)=>{
    return <div {...props}>
        {props.children}
        <button id={"expandButton"}  onClick={props.expand}>{props.className === "contentBlock" ? ">" : "<"}</button>
    </div>
}

const LOAD_COUNT= 3;

export default function ProjectPage(props){
    let {dataset} = useParams()

    let [rawExpanded, setRawExpanded] = useState(false)
    let [preprocessedExpanded, setPreprocessedExpanded] = useState(false)
    let [experimentsExpanded, setExperimentsExpanded] = useState(false)

    let project = useProject(dataset)
    if(project === undefined){
        return <div>Loading...</div>
    }
    return <>
        <Navbar/>
        <div className="parent">
            <ContentBlock className={rawExpanded ? "contentBlockE" : "contentBlock"} expand={()=>setRawExpanded(!rawExpanded)}>
                <ImageView project={project} load_count={LOAD_COUNT} preprocessed={false}>
                    <p style={{"text-align": "center"}}>Raw Images</p>
                </ImageView>
            </ContentBlock>
            <ContentBlock className={preprocessedExpanded ? "contentBlockE" : "contentBlock"} expand={()=>setPreprocessedExpanded(!preprocessedExpanded)}>
                <ImageView project={project} load_count={LOAD_COUNT} preprocessed={true}>
                    <p style={{"text-align": "center"}}>Preprocessed Images</p>
                </ImageView>
            </ContentBlock>
            <ContentBlock className={experimentsExpanded ? "contentBlockE" : "contentBlock"} expand={()=>setExperimentsExpanded(!experimentsExpanded)}>
                <Experiments project={project}>
                    <p style={{"text-align": "center"}}>Experiments</p>
                </Experiments>
            </ContentBlock>
        </div>
    </>
}