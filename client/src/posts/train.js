

export function postTrainExperiment(project, name, fold, extension, model){
    //@app.route("/train/<project_id>/<experiment_name>/<fold>/<model>/<extension_name>", methods=["POST"])
    if(name === "" || fold === ""){
        throw new Error("Name and Fold are required")
        return
    }
    if (model === ""){
        model = "None"
    }
    if (extension === ""){
        extension = "None"
    }

    fetch(`http://localhost:3001/train/${project}/${name}/${fold}/${model}/${extension}`, {
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
