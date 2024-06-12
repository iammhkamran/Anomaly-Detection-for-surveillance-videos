import React from 'react'
import { Link } from 'react-router-dom'

const Sidebar = () => {
    return (
        <div className='sidebar'>
            <h1 className='sidebar--heading'>Anomaly Detection For Surveillance Videos</h1>
            <div className='sidebar--links--container'>
                <Link to='/' className="sidebar--links">Recorded Video</Link>
                <Link to='/Live-Classification' className="sidebar--links">Live Classification</Link>
                {/* <Link to='/Logs' className="sidebar--links">Logs</Link> */}
            </div>
        </div>
    )
}

export default Sidebar