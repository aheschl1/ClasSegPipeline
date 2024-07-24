import './style.css'
import {useParams} from "react-router-dom";
import {useProject} from "../../hooks/projects";
import {useExperiment} from "../../hooks/experiments";
import Navbar from "../../components/navbar/navbar";
import {useEffect} from "react";

export default function ProjectPage(props) {
    let {experiment, dataset} = useParams()
    let experiment_data = useExperiment(dataset, experiment)

    useEffect(() => {
        const configBox = document.getElementById("configBox");
        const scrollableBox = document.getElementById("scrollableBox");
        function matchHeights() {
            scrollableBox.style.height = configBox.offsetHeight + "px";
        }
        matchHeights();
        window.addEventListener("resize", matchHeights);
        return () => window.removeEventListener("resize", matchHeights);
    }, [experiment_data]);

    return <>
        <Navbar/>
        <p>{experiment_data.name}</p>
        <div className="topRow">
            <div className="textBox" id="configBox">
                <h3 style={{textAlign: "center"}}>Config</h3>
                <pre>{JSON.stringify(experiment_data.config, null, 4)}</pre>
            </div>
            <div className="textBoxScrollable" id="scrollableBox">
                <h3 style={{textAlign: "center"}}>Logs</h3>
                <pre>{experiment_data.logs}</pre>
            </div>
        </div>
    </>
}