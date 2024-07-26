import './style.css'
import {useParams} from "react-router-dom";
import {useProject} from "../../hooks/projects";
import {useExperiment} from "../../hooks/experiments";
import Navbar from "../../components/navbar/navbar";
import {useEffect} from "react";
import {Line} from "react-chartjs-2";
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

export default function ExperimentPage(props) {
    let {experiment, dataset} = useParams()
    let experiment_data = useExperiment(dataset, experiment)

    useEffect(() => {
        const configBox = document.getElementById("configBox");
        const scrollableBox = document.getElementById("scrollableBox");
        if (configBox === null || scrollableBox === null) {
            return;
        }
        function matchHeights() {
            scrollableBox.style.height = configBox.offsetHeight + "px";
        }
        matchHeights();
        window.addEventListener("resize", matchHeights);
        return () => window.removeEventListener("resize", matchHeights);
    }, [experiment_data]);
    if (experiment_data === undefined) {
        return <>Loading...</>
    }
    let trainChartData = {
        labels: experiment_data.train_losses.map((_, i) => i),
        datasets: [{
            label: "Train Loss",
            data: experiment_data.train_losses,
            fill: false,
            borderColor: "rgb(75, 192, 192)",
            tension: 0.1
        },
        {
            label: "Val Loss",
            data: experiment_data.val_losses,
            fill: false,
            borderColor: "rgb(192, 75, 75)",
            tension: 0.1
        }]
    }

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
        <div className="topRow">
            <div className="textBox">
                <h3 style={{textAlign: "center"}}>Info</h3>
                <pre>{JSON.stringify({
                    "Name": experiment_data.name,
                    "Dataset": experiment_data.dataset,
                    "Fold": experiment_data.fold,
                    "Available Checkpoints": experiment_data.checkpoints,
                    "Model Kwargs": experiment_data.model_kwargs,
                    "Params": experiment_data.total_params,
                    "Untrainable Params": experiment_data.untrainable_params,
                    "Best Epoch": experiment_data.best_epoch,
                    "Latest Epoch": experiment_data.latest_epoch,
                    "Best Loss": experiment_data.best_loss,
                    "Mean Epoch Time (s)": experiment_data.mean_time
                }, null, 4)}</pre>
            </div>
        </div>
        <Line className="chart" data={trainChartData}></Line>
    </>
}