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

const LOAD_COUNT= 6;

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
            <ContentBlock className={rawExpanded ? "contentBlockE" : "contentBlock"} expand={()=> {
                setRawExpanded(!rawExpanded)
                setPreprocessedExpanded(false)
                setExperimentsExpanded(false)
            }}>
                <ImageView project={project} load_count={!rawExpanded ? LOAD_COUNT : LOAD_COUNT*2} preprocessed={false} expanded={rawExpanded}>
                    <p style={{"text-align": "center"}}>Raw Images</p>
                </ImageView>
            </ContentBlock>
            <ContentBlock className={preprocessedExpanded ? "contentBlockE" : "contentBlock"} expand={()=> {
                setPreprocessedExpanded(!preprocessedExpanded);
                setRawExpanded(false);
                setExperimentsExpanded(false)
            }}>
                <ImageView project={project} load_count={!preprocessedExpanded ? LOAD_COUNT : LOAD_COUNT*2} preprocessed={true} expanded={preprocessedExpanded}>
                    <p style={{"text-align": "center"}}>Preprocessed Images</p>
                </ImageView>
            </ContentBlock>
            <ContentBlock className={experimentsExpanded ? "contentBlockE" : "contentBlock"} expand={()=> {
                setExperimentsExpanded(!experimentsExpanded)
                setRawExpanded(false)
                setPreprocessedExpanded(false)
            }}>
                <Experiments project={project}>
                    <p style={{"text-align": "center"}}>Experiments</p>
                </Experiments>
            </ContentBlock>
        </div>
    </>
}