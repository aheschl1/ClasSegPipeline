import { useState, useEffect } from 'react';

export function useProjects() {
    const [projects, setProjects] = useState([]);

    useEffect(() => {
        fetch('http://localhost:3001/projects')
            .then(response => response.json())
            .then(data => setProjects(data))
            .catch(error => console.error('Error:', error));
    }, []);

    return projects;
}