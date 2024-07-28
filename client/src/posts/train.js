

export function postTrainExperiment(project, name, fold, extension, model, config){
    if(name === "" || fold === ""){
        throw new Error("Name and Fold are required")
    }
    if (model === ""){
        model = "None"
    }
    if (extension === ""){
        extension = "None"
    }

    fetch(`http://localhost:3001/train/${project}/${name}/${fold}/${model}/${extension}/${config}`, {
        method: "POST",
    }).then((response)=>{
        if(response.status === 200){
            return response.json()
        }else{
            console.log(response)
            alert("Error: " + response.status)
        }
    })
}
