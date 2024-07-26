import {useEffect, useState} from "react";


export function useReadme(){
    let [readme, setReadme] = useState("")
    useEffect(()=>{
        fetch("http://localhost:3001/README.md")
            .then(response => response.text())
            .then(data => setReadme(data))
            .catch(error => console.error('Error:', error));
    }, []);
    return readme;
}