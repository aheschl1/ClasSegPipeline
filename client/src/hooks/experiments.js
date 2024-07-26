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

export function useExperiment(dataset, experiment){
    let [experimentData, setExperimentData] = useState(undefined)

    useEffect(()=>{
        fetch(`http://localhost:3001/projects/${dataset}/experiments/${experiment}`)
            .then(response => response.json())
            .then(data => setExperimentData(data))
            .catch(error => console.error('Error:', error));
    }, [dataset, experiment]);

    return experimentData;
}