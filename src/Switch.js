class Switch extends React.Component {
    render() {
        return (
            <div className="form-check form-switch w-50 col"
                 style={{marginTop: '50px',
                     marginBottom: '15px'}}>
                <input className="form-check-input" type="checkbox" role="switch" id={this.props.id} onClick={this.props.handleClick}/>
                    <label className="form-check-label" htmlFor="flexSwitchCheckDefault">{this.props.text}</label>
            </div>
        )
    }
}