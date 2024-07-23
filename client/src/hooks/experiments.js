import {useState, useEffect} from "react";

export function useExperiments(project){
    let [experiments, setExperiments] = useState([])

    useEffect(()=>{
        fetch(`http://localhost:3001/projects/${project.name}/experiments`)
            .then(response => response.json())
            .then(data => setExperiments(data))
            .catch(error => console.error('Error:', error));
    }, [project]);

    return experiments;
}

export function useExperiment(project, experiment){
    let [experimentData, setExperimentData] = useState({
        name: "",
        fold: "",
        checkpoints: []
    })

    useEffect(()=>{
        if (project === undefined){
            return
        }
        fetch(`http://localhost:3001/projects/${project.name}/experiments/${experiment.name}`)
            .then(response => response.json())
            .then(data => setExperimentData(data))
            .catch(error => console.error('Error:', error));
    }, [project, experiment]);

    return experimentData;
}