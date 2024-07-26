import {useProject} from '../../hooks/projects'
import {useParams} from 'react-router-dom'
import './style.css'
import Navbar from "../../components/navbar/navbar";
import ImageView from "./image_views/image_view";
import Experiments from "./experiments_view/experiments";
import {useState} from "react";
import {postTrainExperiment} from "../../posts/train";


const ContentBlock = (props)=>{
    return <div {...props}>
        {props.children}
        <button id={"expandButton"}  onClick={props.expand}>{props.className === "contentBlock" ? ">" : "<"}</button>
    </div>
}

function trainExperiment(project, name, fold, extension, model){
    if(name === "" || fold === ""){
        alert("Name and Fold are required")
        return
    }
    postTrainExperiment(project, name, fold, extension, model)
}

const LOAD_COUNT= 6;

export default function ProjectPage(props){
    let {dataset} = useParams()

    let [rawExpanded, setRawExpanded] = useState(false)
    let [preprocessedExpanded, setPreprocessedExpanded] = useState(false)
    let [experimentsExpanded, setExperimentsExpanded] = useState(false)
    let [buildingExperiment, setBuildingExperiment] = useState(false)

    let project = useProject(dataset)

    const newExperiment = ()=>{
        setBuildingExperiment(!buildingExperiment)
    }

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
                    <p style={{"textAlign": "center"}}>Raw Images</p>
                </ImageView>
            </ContentBlock>
            <ContentBlock className={preprocessedExpanded ? "contentBlockE" : "contentBlock"} expand={()=> {
                setPreprocessedExpanded(!preprocessedExpanded);
                setRawExpanded(false);
                setExperimentsExpanded(false)
            }}>
                <ImageView project={project} load_count={!preprocessedExpanded ? LOAD_COUNT : LOAD_COUNT*2} preprocessed={true} expanded={preprocessedExpanded}>
                    <p style={{"textAlign": "center"}}>Preprocessed Images</p>
                </ImageView>
            </ContentBlock>
            <ContentBlock className={experimentsExpanded ? "contentBlockE" : "contentBlock"} expand={()=> {
                setExperimentsExpanded(!experimentsExpanded)
                setRawExpanded(false)
                setPreprocessedExpanded(false)
            }}>
                <Experiments project={project}>
                    <div>
                        <div style={{
                            "display": "flex",
                            "justifyContent": "space-between",
                            "padding": "10px"
                        }}>
                            <p style={{"textAlign": "center"}}>Experiments</p>
                            <button onClick={newExperiment}>{buildingExperiment ? "Cancel" : "New Experiment"}</button>
                        </div>
                        <div style={{
                            "visibility": buildingExperiment ? "visible" : "hidden",
                            "height": buildingExperiment ? "auto" : "0",
                            "width": buildingExperiment ? "auto" : "0",
                            "padding": buildingExperiment ? "10px" : "0",
                        }}>
                            <div style={{
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr"
                            }}>
                                <input id="experimentNameInput" type="text" placeholder={"Experiment Name (Required)"}/>
                                <input id="experimentFold" type="number" placeholder={"Fold (Required)"}/>
                                <input id="experimentExtension" type="text" placeholder={"Extension (Default)"}/>
                                <input id="experimentModel" type="text" placeholder={"Model (None)"}/>
                                <button
                                    onClick={() => trainExperiment(
                                        project.name,
                                        document.getElementById("experimentNameInput").value,
                                        document.getElementById("experimentFold").value,
                                        document.getElementById("experimentExtension").value,
                                        document.getElementById("experimentModel").value
                                    )}
                                >Train
                                </button>
                            </div>
                        </div>
                    </div>
                </Experiments>
            </ContentBlock>
        </div>
    </>
}