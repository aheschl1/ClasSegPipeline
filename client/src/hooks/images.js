import {useEffect, useState} from "react";

export function useImage(project, preprocessed, case_number){
    let [image, setImage] = useState(undefined)
    useEffect(()=>{
        fetch(`'http://localhost:3001/projects/${project}/${preprocessed?"preprocessed":"raw"}/${case_number}`)
            .then(res=>res.json()).then(data=>{
                setImage(data)
            })
    }, [project, preprocessed, case_number])
    return image
}