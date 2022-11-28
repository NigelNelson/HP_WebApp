class SiftButton extends React.Component {
    render() {
        return (
            <button type="button"
                    className="btn btn-primary w-50 border border-dark"
                    onClick={this.props.onClick}
                    style={{maxHWidth: '200px',
                        marginTop: '10px',
                        marginBottom: '15px '}}
                    id={this.props.id}>
                {this.props.isRunning &&
                    <span className="spinner-border spinner-border-sm m-1"></span>
                }
                {this.props.isRunning &&
                   <div className="m-0 p-0">Computing...</div>
                }
                {!this.props.isRunning &&
                    <div className="m-0 p-0">Get Sift Points</div>
                }
            </button>
        )
    }
}